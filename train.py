#!/usr/bin/env python
"""Train a radiance field with nerfstudio."""

import random
import torch
import numpy as np
import traceback
from datetime import timedelta
from typing import Any, Callable, Optional
import yaml
from rich.console import Console
import tyro

from gssr.configs import base_config as cfg
from gssr.engine.trainer import Trainer

CONSOLE = Console(width=120)
DEFAULT_TIMEOUT = timedelta(minutes=30)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore
torch.set_float32_matmul_precision("high")


def _set_random_seed(seed) -> None:
    """Set randomness seed in torch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_loop(local_rank: int, world_size: int, config: cfg.Config, global_rank: int = 0):
    """Main training function that sets up and runs the trainer per process

    Args:
        local_rank: current rank of process
        world_size: total number of gpus available
        config: config file specifying training regimen
    """
    _set_random_seed(config.machine.seed + global_rank)
    torch.cuda.set_device(local_rank)
    trainer = Trainer(config, local_rank, world_size)
    trainer.setup()
    trainer.train()


def launch(
    main_func: Callable,
    num_gpus_per_machine: int,
    num_machines: int = 1,
    machine_rank: int = 0,
    dist_url: str = "auto",
    config: Optional[cfg.Config] = None,
    timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
    """Function that spawns muliple processes to call on main_func

    Args:
        main_func (Callable): function that will be called by the distributed workers
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int, optional): total number of machines
        machine_rank (int, optional): rank of this machine.
        dist_url (str, optional): url to connect to for distributed jobs.
        config (Config, optional): config file specifying training regimen.
        timeout (timedelta, optional): timeout of the distributed workers.
    """
    assert config is not None
    world_size = num_machines * num_gpus_per_machine
    if world_size <= 1:
        # world_size=0 uses one CPU in one process.
        # world_size=1 uses one GPU in one process.
        try:
            main_func(local_rank=0, world_size=world_size, config=config)
        except KeyboardInterrupt:
            # print the stack trace
            CONSOLE.print(traceback.format_exc())

    elif world_size > 1:
        # Using multiple gpus with multiple processes.
        print("Not support!")


def main(config: cfg.Config) -> None:
    """Main function."""

    config.set_timestamp()

    if config.trainer.load_config:
        CONSOLE.log(f"Loading pre-set config from: {config.trainer.load_config}")
        config = yaml.load(config.trainer.load_config.read_text(), Loader=yaml.Loader)

    # print and save config
    config.print_to_terminal()
    config.save_config()

    launch(
        main_func=train_loop,
        num_gpus_per_machine=config.machine.num_gpus,
        num_machines=config.machine.num_machines,
        machine_rank=config.machine.machine_rank,
        dist_url=config.machine.dist_url,
        config=config,
    )


from gssr.configs.method_config import AnnotatedBaseConfigUnion
from gssr.configs.config_utils import convert_markup_to_ansi
from gssr.configs.method_config import method_configs

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    # Choose a base configuration and override values.
    tyro.extras.set_accent_color("bright_yellow")
    main(
        tyro.cli(
            AnnotatedBaseConfigUnion,
            description=convert_markup_to_ansi(__doc__),
        )
    )

    # ## for test
    # main(method_configs['octree-pgsr'])

if __name__ == "__main__":
    entrypoint()

