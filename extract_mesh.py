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

from gssr.configs import base_config as cfg
from gssr.scene.base_scene import Scene
from gssr.utils.mesh_utils import GaussianExtractor, post_process_mesh
from gssr.utils.render_utils import generate_path, create_videos

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


def eval_setup(config_path: Path) -> Tuple[cfg.Config, Scene, Path]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    config.trainer.load_gaussian_dir = config.get_gaussian_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup scene (which includes the dataloader and gaussians)
    scene = config.scene.setup(source_dir = config.source_path, eval = False, device = device)
    assert isinstance(scene, Scene)

    # load gaussians information
    gaussian_path = eval_load_gaussians(config.trainer, scene)

    return config, scene, gaussian_path


@dataclass
class MeshExtractor:
    """Load a gaussian-model, extract mesh"""

    # Path to config YAML file.
    load_config: Path
    skip_train: bool = False
    skip_test: bool = False
    skip_mesh: bool = False
    render_video: bool = False
    frames: int = 240
    unbounded: bool = False
    depth_trunc: float = -1
    voxel_size: float = -1
    sdf_trunc: float = -1
    num_cluster: int = 50
    mesh_res: int = 1024

    def main(self) -> None:
        """Main function."""
        config, scene, _ = eval_setup(config_path=self.load_config)
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
    
        if self.render_video:
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
    
def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MeshExtractor).main()

if __name__ == "__main__":
    entrypoint()
