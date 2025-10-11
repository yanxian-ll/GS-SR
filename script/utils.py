import torch
from pathlib import Path
from typing import Optional, Tuple, List
import yaml
import os
import sys
from rich.console import Console

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from gssr.configs import base_config as cfg
from gssr.scene.base_scene import Scene

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

def eval_setup(config_path: Path, iterations: Optional[int] = None, data_device: str = "cuda") -> Tuple[cfg.Config, Scene, Path]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    config.trainer.load_gaussian_dir = config.get_gaussian_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup scene (which includes the dataloader and gaussians)
    config.scene.dataloader.device = data_device
    scene = config.scene.setup(source_dir = config.source_path, eval = config.eval, device = device)
    assert isinstance(scene, Scene)

    # load gaussians information
    if iterations is not None:
        config.trainer.load_gaussian_step = iterations
    gaussian_path = eval_load_gaussians(config.trainer, scene)
    return config, scene, gaussian_path

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
