from pathlib import Path
import os
import sys
import torch
import tyro
import numpy as np
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from gssr.cameras import OrthoCamera
from gssr.utils.render_utils import save_img_f32, save_img_u8, save_vis_depth
from gssr.utils.graphics_utils import focal2fov
from gssr.utils.partition_utils import get_axis_aligned_bounding_box, split_points_tile
from utils import eval_setup

CONSOLE = Console(width=120)

@dataclass
class OrthoRender:
    """Load a gaussian-model, extract mesh"""

    # Path to config YAML file.
    load_config: Path = Path()

    # iterations of gaussians to load, if None, load the latest one
    iterations: Optional[int] = None

    # Minimum Number of Gaussians per Tile（为了剔除在边缘的tile）
    min_points: int = 0
    # Overlap Ratio Between Adjacent Tiles
    extent_ratio: float = 1.05

    # Ground Sample Distance (m)
    gsd: Optional[float] = None
    # tile size (m)
    tile_size: Optional[float] = None
    # camera height (m)
    camera_height: Optional[float] = None

    # data device
    data_device: str = "cuda"

    @torch.no_grad()
    def generate_cameras(self, points: np.ndarray, camera_height: float, gsd:float, tile_size: float, min_points: int, extent_ratio: float):
        # get bounding box
        bbx, _ = get_axis_aligned_bounding_box(points)
        # partition
        list_tiles = split_points_tile(points, property=None, bbx=bbx, 
                                       tile_size=tile_size, 
                                       min_points=min_points, 
                                       extent_ratio=extent_ratio)
        print(f"Split the Scene into {len(list_tiles)} Tiles")
        if len(list_tiles) <= 0:
            raise ValueError

        cameras = []
        for i, tile in enumerate(list_tiles):
            bbx = tile['bbx']     #(4,2)
            mx, Mx, my, My = np.min(bbx[:,0]), np.max(bbx[:,0]), np.min(bbx[:,1]), np.max(bbx[:,1])
            ground_width, ground_height = Mx-mx, My-my
            image_width = int(ground_width / gsd + 0.5)
            image_height = int(ground_height / gsd + 0.5)
            if i==0:
                print(f"Ground_width: {ground_width}, Ground_height: {ground_height}, Width: {image_width}, Height: {image_height}")

            c_x = (bbx[0,0] + bbx[2,0]) / 2.0
            c_y = (bbx[0,1] + bbx[2,1]) / 2.0
            c_h = camera_height
            c2w = np.array([
                [1, 0, 0, c_x],
                [0, 1, 0, c_y],
                [0, 0, -1, c_h],
                [0, 0, 0, 1]
            ])

            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3].T
            T = w2c[:3, -1]

            fx = c_h / gsd
            fy = c_h / gsd

            FovX = focal2fov(fx, image_width)
            FovY = focal2fov(fy, image_height)
            gt_image = torch.ones((3, image_height, image_width), dtype=torch.float32)
            loaded_mask = None
            image_name = f"tile_{tile['tile_id']}"

            ground_width = Mx - mx
            ground_height = My - my

            camera_id = 50
            ortho_cam = OrthoCamera(
                colmap_id=camera_id, R=R, T=T,
                FoVx=FovX, FoVy=FovY,
                ground_width=ground_width, ground_height=ground_height,
                image=gt_image, gt_alpha_mask=loaded_mask,
                image_name=image_name, uid=camera_id,
                data_device=self.data_device
            )
            setattr(ortho_cam, "bbx", [mx, Mx, my, My])
            cameras.append(ortho_cam)
        return cameras

    def calculate_gsd(self, scene):
        train_dataset = scene.dataloader.getTrainData()
        scale = scene.dataloader.config.scene_scale
        points = scene._gaussians._xyz.detach().cpu().numpy()

        list_camera_height = []
        list_image_size = []
        list_camera_focal = []
        for cam in train_dataset:
            camera_height = cam.camera_center[-1].detach().cpu().item()
            list_camera_height.append(camera_height)
            list_image_size.append(cam.image_width)
            list_image_size.append(cam.image_height)
            list_camera_focal.append(cam.Fx)
            list_camera_focal.append(cam.Fy)
        mean_camera_height = sum(list_camera_height) / len(list_camera_height)
        mean_image_size = int(sum(list_image_size) / len(list_image_size))
        mean_camera_focal = int(sum(list_camera_focal) / len(list_camera_focal))

        ## estimate tile-size and gsd
        h = mean_camera_height - np.mean(points[:, -1])
        tile_size = h / mean_camera_focal * mean_image_size
        tile_size *= scale
        gsd = h / mean_camera_focal
        gsd *= scale
        mean_camera_height *= scale
        return gsd, tile_size, mean_camera_height

    @torch.no_grad()
    def main(self, load_config=None, save=True)-> Tuple[Optional[list], Optional[list], Optional[list], Optional[tuple]]:
        """Main function."""
        config, scene, _ = eval_setup(config_path=load_config if load_config else self.load_config, iterations=self.iterations, data_device=self.data_device)
        points = scene._gaussians._xyz.detach().cpu().numpy()
        scale = scene.dataloader.config.scene_scale
        tx = scene.dataloader.config.t_x
        ty = scene.dataloader.config.t_y
        tz = scene.dataloader.config.t_z

        gsd, tile_size, camera_height = self.calculate_gsd(scene)

        if self.tile_size is None:
            self.tile_size = tile_size
        if self.gsd is None:
            self.gsd = gsd
        if self.camera_height is None:
            self.camera_height = camera_height
        print(f"Estimated GSD: {gsd}, Tile_size: {tile_size}, Camera_height: {camera_height}")

        ## generate cameras
        cameras = self.generate_cameras(points, camera_height=camera_height / scale, gsd=self.gsd / scale, tile_size=self.tile_size / scale, min_points=self.min_points, extent_ratio=self.extent_ratio)

        ## rendering
        list_render = []
        list_depth = []
        for idx, viewpoint_cam in tqdm(enumerate(cameras), desc="reconstruct radiance fields"):
            means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, _ = scene.generate_ortho_gaussians(viewpoint_cam)
            render_pkg = scene.render_ortho(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)

            list_render.append(render_pkg['render'].permute(1,2,0).detach().cpu().numpy())  #(h,w,3)
            if 'depth' in render_pkg:
                list_depth.append(render_pkg['depth'][0].detach().cpu().numpy())  #(h,w)

        if save:
            out_dir = os.path.join(config.get_base_dir(), 'ortho', "ours_{}".format(config.trainer.load_gaussian_step))
            render_path = os.path.join(out_dir, "renders")
            vis_path = os.path.join(out_dir, "depths")
            coordinate_path = os.path.join(out_dir, "coordinate")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(vis_path, exist_ok=True)
            os.makedirs(coordinate_path, exist_ok=True)

            ## Save
            for idx, viewpoint_cam in tqdm(enumerate(cameras), desc="save render and depth"):
                save_img_u8(list_render[idx], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

                if len(list_depth) > 0:
                    depth = (self.camera_height - list_depth[idx]) * scale + tz
                    save_img_f32(depth, os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
                    
                    distance_limits = np.percentile(depth.flatten(), [3, 100 - 3])
                    lo, hi = [x for x in distance_limits]
                    save_vis_depth(depth, lo, hi, os.path.join(vis_path, 'depth_vis_{0:05d}'.format(idx) + ".png"))

                with open(os.path.join(coordinate_path, '{0:05d}'.format(idx) + ".txt"), 'w') as f:
                    f.write(f"{viewpoint_cam.image_width}\n")
                    f.write(f"{viewpoint_cam.image_height}\n")
                    f.write(f"{self.gsd}\n")
                    f.write(f"{viewpoint_cam.bbx[0] * scale + tx}\n")
                    f.write(f"{viewpoint_cam.bbx[2] * scale + ty}\n")

        return cameras, list_render, list_depth, (scale, tx, ty, tz)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(OrthoRender).main()

if __name__ == "__main__":
    entrypoint()
