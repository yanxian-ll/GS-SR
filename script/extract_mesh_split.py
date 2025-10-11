from pathlib import Path
import tyro
from dataclasses import dataclass
from rich.console import Console
import os
import numpy as np
import open3d as o3d

from extract_mesh import MeshExtractor
from utils import get_tile_configs

CONSOLE = Console(width=120)


@dataclass
class MeshSplitExtractor(MeshExtractor):
    # Path to config YAML file.
    load_config: Path = None

    def main_(self) -> None:
        config, tile_configs = get_tile_configs(self.load_config)

        train_dir = os.path.join(config.get_base_dir(), 'train', 'our_mesh')
        os.makedirs(train_dir, exist_ok=True)

        for i, load_tile_config in enumerate(tile_configs):
            tile_config, mesh = self.main(load_tile_config)
            
            # crop
            with open(os.path.join(tile_config.source_path, "box.txt"), 'r') as f:
                f.readline()
                mx, Mx, my, My = [float(item) for item in f.readline().strip().split(" ")]

            points = np.array(mesh.vertices)
            mz, Mz = np.min(points[:,-1]), np.max(points[:,-1])
            mz, Mz = mz-(Mz-mz) * 0.1, Mz+(Mz-mz) * 0.1

            bounding_box = o3d.utility.Vector3dVector(np.array(
                [[mx, my, mz],
                [Mx, my, mz],
                [Mx, My, Mz],
                [mx, My, mz]]))
            
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(bounding_box)
            cropped_mesh = mesh.crop(obb)
            o3d.io.write_triangle_mesh(os.path.join(train_dir, "tile_%04d.ply" % i), cropped_mesh)

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MeshSplitExtractor).main_()

if __name__ == "__main__":
    entrypoint()
