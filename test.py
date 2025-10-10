from pathlib import Path
import yaml
import os
import sys
import torch
import tyro
import numpy as np
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple, Optional
from tqdm import tqdm
import rasterio
from rasterio.transform import from_origin

from gssr.configs import base_config as cfg
from gssr.cameras import SunCamera2
from gssr.scene.base_scene import Scene
from gssr.utils.render_utils import save_img_f32, save_img_u8, save_vis_depth
from gssr.utils.graphics_utils import focal2fov
from gssr.utils.partition_utils import get_axis_aligned_bounding_box, split_points_tile


from plyfile import PlyData, PlyElement

CONSOLE = Console(width=120)

def eval_load_gaussians(config: cfg.TrainerConfig, scene: Scene) -> Path:
    assert config.load_gaussian_dir is not None
    if config.load_gaussian_step is None:
        CONSOLE.log(f"Loading latest gaussians from {config.load_gaussian_dir}")
        if not os.path.exists(config.load_gaussian_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No gaussians directory found at {config.load_gaussian_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the gaussians exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        else:
            load_step = max([int(x[x.find("_") + 1 : x.find(".")]) for x in os.listdir(config.load_gaussian_dir) if x.endswith('.ply')])
            config.load_gaussian_step = load_step
    else:
        load_step = config.load_gaussian_step
    
    load_path = config.load_gaussian_dir / f"iteration_{load_step}.ply"
    scene._gaussians.load_gaussians(load_path)
    scene._gaussians.load_mlp_checkpoints(config.load_gaussian_dir)
    CONSOLE.print(f":white_check_mark: Done loading gaussians from {load_path}")
    return load_path


def eval_setup(config_path: Path, data_device: str = "cuda") -> Tuple[cfg.Config, Scene, Path]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    config.trainer.load_gaussian_dir = config.get_gaussian_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup scene (which includes the dataloader and gaussians)
    config.scene.dataloader.device = data_device
    scene = config.scene.setup(source_dir = config.source_path, eval = False, device = device)
    assert isinstance(scene, Scene)

    # load gaussians information
    gaussian_path = eval_load_gaussians(config.trainer, scene)

    return config, scene, gaussian_path


def write_georeference(image_path, output_path, ulx, uly, pixel_size):
    with rasterio.open(image_path) as src:
        profile = src.profile

        # 更新地理参考信息
        transform = from_origin(ulx, uly, pixel_size, -pixel_size)
        profile.update({
            'transform': transform,
            'crs': 'EPSG:4326'  # 假设使用 WGS84 坐标系
        })

    # 写入新的影像文件
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(src.read())


@dataclass
class OrthoRender:
    """Load a gaussian-model, extract mesh"""

    # Path to config YAML file.
    load_config: Path = None

    # data device
    data_device: str = "cuda"

    @torch.no_grad()
    def main(self, load_config=None):
        """Main function."""
        config, scene, _ = eval_setup(config_path=load_config if load_config else self.load_config)
        cameras = scene.dataloader.getTrainData()
        points = scene._gaussians._xyz.detach().cpu().numpy()

        ortho_cameras = []
        for i, cam in enumerate(cameras):
            R = cam.R
            T = cam.T
            K = cam.get_k().cpu().numpy()

            points_ic = (R.T @ points.T).T + T.reshape(1,3)
            mean_h = np.percentile(points_ic[:, -1], q=20)
            focal = (cam.Fx + cam.Fy) / 2.0
            gsd = mean_h / focal

            uvs = (K @ points_ic.T).T
            uvs = uvs[:, :2] / uvs[:, 2:3]
            w, h = cam.image_width, cam.image_height
            mask = (uvs[:,0]>=0) & (uvs[:,0]<w) & (uvs[:,1]>=0) & (uvs[:,1]<h)
            points_ic = points_ic[mask]
            dx = np.max(points_ic[:,0]) - np.min(points_ic[:,0])
            dy = np.max(points_ic[:,1]) - np.min(points_ic[:,1])

            image_width = int(dx / gsd + 0.5)
            image_height = int(dy / gsd + 0.5)

            if image_width > 1600:
                image_width = 1600
                dx = image_width * gsd
            if image_height > 1600:
                image_height = 1600
                dy = image_height * gsd

            FovX = focal2fov(focal, image_width)
            FovY = focal2fov(focal, image_height)
            gt_image = torch.ones((3, image_height, image_width), dtype=float)
            loaded_mask = None
            image_name = f"view_{i}"

            # camera_id = 0
            # cam.sun_camera = SunCamera2(colmap_id=camera_id, R=R, T=T, 
            #             FoVx=FovX, FoVy=FovY, dx=dx, dy=dy,
            #             image=gt_image, gt_alpha_mask=loaded_mask,
            #             image_name=image_name, uid=camera_id, data_device=self.data_device)
            ortho_cameras.append(cam)

        ## rendering
        for idx, viewpoint_cam in tqdm(enumerate(ortho_cameras[:5]), desc="reconstruct radiance fields"):
            means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, _ = scene.generate_gaussians(viewpoint_cam)
            # render_pkg = scene.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
            # simp_render_pkg = scene.simp_render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
            simp_render_pkg = scene.simp_render_ortho(ortho_cameras[idx+5].sun_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp[:,:3])

            weights = simp_render_pkg['weights'].reshape(-1,1)
            # print(weights.max(), weights.min())

            counts = simp_render_pkg['counts'].reshape(-1,1)
            weights[counts <= 0] = 0.0
            weights[counts > 0] = weights[counts > 0] / counts[counts > 0]
            # print(weights.max(), weights.min())

            # weights = torch.nn.functional.sigmoid(weights)
            # weights = torch.nn.functional.softmax(weights, dim=0)
            weights = torch.clip(weights, 0.0, 1.0)
            q = torch.quantile(weights, q=0.99)
            weights[weights>q] = weights.max()
            weights = weights / weights.max()
            # print(weights.max(), weights.min())
            # print((weights>q).sum()/(weights>=0).sum())

            colors_precomp = torch.concatenate([colors_precomp, weights], dim=-1)
            render_pkg = scene.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)

            render = simp_render_pkg['render'][:3,:,:].permute(1,2,0).detach().cpu().numpy()
            save_img_u8(render, os.path.join('./test_output', '{0:05d}'.format(idx) + "_ortho.png"))

            render = render_pkg['render'][:3,:,:].permute(1,2,0).detach().cpu().numpy()
            save_img_u8(render, os.path.join('./test_output', '{0:05d}'.format(idx) + "_render.png"))

            shadow = render_pkg["render"][7:8, :, :]
            shadow = (shadow.squeeze().detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
            # print(shadow.min(), shadow.max())
            import cv2
            shadow_show = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)
            save_img_u8(shadow_show, os.path.join('./test_output', '{0:05d}'.format(idx) + "_shadow.png"))

            weights = simp_render_pkg['weights'].detach().cpu().numpy().reshape(-1,1)
            counts = simp_render_pkg['counts'].detach().cpu().numpy().reshape(-1,1)

            # save ply
            xyz = means3D.detach().cpu().numpy()
            opacities = opacity.detach().cpu().numpy()
            weights[counts <= 0] = 0.0
            weights[counts > 0] = weights[counts > 0] / counts[counts > 0]
            
            dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4'), ('weight', 'f4')]
            elements = np.empty(xyz.shape[0], dtype=dtype_full)
            attributes = np.concatenate((xyz, opacities, weights), axis=1)
            elements[:] = list(map(tuple, attributes))
            el = PlyElement.describe(elements, 'vertex')
            PlyData([el]).write(os.path.join("./test_output", '{0:05d}'.format(idx) + ".ply"))


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(OrthoRender).main()


if __name__ == "__main__":
    entrypoint()
