import os
import tyro
import copy
from pathlib import Path
import train
from train import AnnotatedBaseConfigUnion, convert_markup_to_ansi, cfg


def main(config: cfg.Config) -> None:
    assert config.partitioner.need_partition, "config.partitioner.need_partition should be True"
    # split scene, TODO
    

    # get tiles
    list_tiles = [os.path.join(config.source_path, t) for t in os.listdir(config.source_path) if t.startswith("tile_")]
    list_configs = [Path("tile_%04d/"%i) for i in range(len(list_tiles))]

    # print and save config
    config.set_timestamp()
    config.print_to_terminal()
    config.partitioner.config_of_tiles = list_configs
    config.save_config()
    
    for i in range(len(list_tiles)):
        tile_config = copy.deepcopy(config)

        # update config
        tile_config.trainer.relative_gaussian_dir = Path(list_configs[i] / config.trainer.relative_gaussian_dir)
        tile_config.trainer.relative_ckpt_dir = Path(list_configs[i]/ config.trainer.relative_ckpt_dir)
        tile_config.relative_config_dir = Path(list_configs[i] / config.relative_config_dir)
        tile_config.relative_log_dir = Path(list_configs[i] / config.relative_log_dir)

        tile_config.source_path = list_tiles[i]
        tile_config.partitioner.need_partition = False
        tile_config.partitioner.config_of_tiles = []

        # train
        train.main(tile_config)


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


if __name__ == "__main__":
    entrypoint()
