from pathlib import Path
import yaml
import os
import sys
import torch
import tyro
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple
import open3d as o3d

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from gssr.utils.mesh_utils import GaussianExtractor, post_process_mesh
from gssr.utils.render_utils import generate_path, create_videos

from utils import eval_setup

CONSOLE = Console(width=120)


@dataclass
class MeshExtractor:
    """Load a gaussian-model, extract mesh"""

    # Path to config YAML file.
    load_config: Path = None
    skip_train: bool = False
    skip_test: bool = False
    skip_mesh: bool = False
    skip_video: bool = False
    frames: int = 240
    
    unbounded: bool = False
    depth_trunc: float = -1
    voxel_size: float = -1
    sdf_trunc: float = -1
    num_cluster: int = 50
    mesh_res: int = 1024

    data_device: str = "cuda"

    def main(self, load_config=None):
        """Main function."""
        config, scene, _ = eval_setup(config_path=load_config if load_config else self.load_config)
        train_cams = scene.dataloader.getTrainData()
        test_cams = scene.dataloader.getTestData()

        ## setup 
        train_dir = os.path.join(config.get_base_dir(), 'train', "ours_{}".format(config.trainer.load_gaussian_step))
        test_dir = os.path.join(config.get_base_dir(), 'test', "ours_{}".format(config.trainer.load_gaussian_step))
        traj_dir = os.path.join(config.get_base_dir(), 'traj', "ours_{}".format(config.trainer.load_gaussian_step))
        gaussExtractor = GaussianExtractor(scene.eval_render)

        if not self.skip_train:
            CONSOLE.log("export training images ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(train_cams)
            gaussExtractor.export_image(train_dir)
        
        if (not self.skip_test) and (len(test_cams) > 0):
            CONSOLE.log("export rendered testing images ...")
            os.makedirs(test_dir, exist_ok=True)
            gaussExtractor.reconstruction(test_cams)
            gaussExtractor.export_image(test_dir)
    
        if not self.skip_video:
            CONSOLE.log("render videos ...")
            os.makedirs(traj_dir, exist_ok=True)
            cam_traj = generate_path(train_cams, n_frames=self.frames)
            gaussExtractor.reconstruction(cam_traj)
            gaussExtractor.export_image(traj_dir)
            create_videos(base_dir=traj_dir, input_dir=traj_dir, out_name='render_traj', num_frames=self.frames)

        if not self.skip_mesh:
            CONSOLE.log("export mesh ...")
            os.makedirs(train_dir, exist_ok=True)
            gaussExtractor.reconstruction(train_cams)
            # extract the mesh and save
            if self.unbounded:
                name = 'fuse_unbounded.ply'
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=self.mesh_res)
            else:
                name = 'fuse.ply'
                depth_trunc = (gaussExtractor.radius * 2.0) if self.depth_trunc < 0  else self.depth_trunc
                voxel_size = (depth_trunc / self.mesh_res) if self.voxel_size < 0 else self.voxel_size
                sdf_trunc = 5.0 * voxel_size if self.sdf_trunc < 0 else self.sdf_trunc
                mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
            
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
            CONSOLE.log("mesh saved at {}".format(os.path.join(train_dir, name)))
            # post-process the mesh and save, saving the largest N clusters
            mesh_post = post_process_mesh(mesh, cluster_to_keep=self.num_cluster)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
            CONSOLE.log("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))
            return config, mesh_post
        else:
            return config, None
    
def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MeshExtractor).main()

if __name__ == "__main__":
    entrypoint()
