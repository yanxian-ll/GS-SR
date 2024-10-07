from pathlib import Path
import tyro
from dataclasses import dataclass
import torch
import yaml
from typing import Tuple, List
from rich.console import Console
import os
import numpy as np
from tqdm import tqdm
import open3d as o3d


from extract_mesh import MeshExtractor, cfg, eval_setup
from gssr.utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh, estimate_bounding_sphere


CONSOLE = Console(width=120)


@torch.no_grad()
def get_tile_configs(config_path: Path) -> Tuple[cfg.Config, List[Path]]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    assert config.partitioner.need_partition, "config.partitioner.need_partition should be True"
    assert len(config.partitioner.config_of_tiles)>0, "please provide config.partitioner.config_of_tiles"

    list_tile_config_path = []
    for idx, cpath in enumerate(config.partitioner.config_of_tiles):
        list_tile_config_path.append(Path(config.get_base_dir() / cpath / "config.yml"))

    return config, list_tile_config_path


@dataclass
class MeshExtractor_(MeshExtractor):
    # Path to config YAML file.
    load_config: Path
    data_device: str = "cpu"

    def main(self) -> None:
        config, tile_configs = get_tile_configs(self.load_config)
        num_tiles = len(tile_configs)
        train_dir = os.path.join(config.get_base_dir(), 'train', 'our_mesh')
        os.makedirs(train_dir, exist_ok=True)

        list_rgbs = []
        list_depths = []
        list_cames = []

        for i in range(num_tiles):
            config, scene, _ = eval_setup(config_path=tile_configs[i], data_device=self.data_device)
            train_cams = scene.dataloader.getMiniTrainData()

            # only render images in bounding-box
            with open(os.path.join(config.source_path, "box.txt"), 'r') as f:
                f.readline()
                mx, Mx, my, My = [float(item) for item in f.readline().strip().split(" ")]
            
            valid_cam = []
            for cam in train_cams:
                center = cam.camera_center.detach().cpu().numpy()
                if (center[0] > mx) and (center[0] < Mx) and (center[1] > my) and (center[1] < My):
                    valid_cam.append(cam)

            ## setup 
            gaussExtractor = GaussianExtractor(scene.eval_render)
            ## reconstruction
            gaussExtractor.reconstruction(valid_cam)
            list_rgbs = list_rgbs + gaussExtractor.rgbmaps
            list_depths = list_depths + gaussExtractor.depthmaps
            list_cames = list_cames + valid_cam
            torch.cuda.empty_cache()

        # setup TSDF-fusion parameter
        radius, center = estimate_bounding_sphere(list_cames)
        depth_trunc = (radius * 2.0) if self.depth_trunc < 0  else self.depth_trunc
        voxel_size = (depth_trunc / self.mesh_res) if self.voxel_size < 0 else self.voxel_size
        sdf_trunc = 5.0 * voxel_size if self.sdf_trunc < 0 else self.sdf_trunc

        CONSOLE.log("Running tsdf volume integration ...")
        CONSOLE.log(f'voxel_size: {voxel_size}')
        CONSOLE.log(f'sdf_trunc: {sdf_trunc}')
        CONSOLE.log(f'depth_truc: {depth_trunc}')

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(list_cames)), desc="TSDF integration progress"):
            rgb = list_rgbs[i]
            depth = list_depths[i]
            
            # if we have mask provided, use it
            if list_cames[i].gt_alpha_mask is not None:
                depth[(list_cames[i].gt_alpha_mask < 0.5)] = 0

            # make open3d rgbd
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),
                depth_trunc = depth_trunc, convert_rgb_to_intensity=False,
                depth_scale = 1.0
            )
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)
        
        name = 'fuse.ply'
        mesh = volume.extract_triangle_mesh()
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        CONSOLE.log("mesh saved at {}".format(os.path.join(train_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh = o3d.io.read_triangle_mesh(os.path.join(train_dir, name))
        mesh_post = post_process_mesh(mesh, cluster_to_keep=self.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        CONSOLE.log("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MeshExtractor_).main()

if __name__ == "__main__":
    entrypoint()
