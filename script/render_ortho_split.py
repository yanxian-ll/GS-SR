import os
from pathlib import Path
from statistics import mean
from typing import Optional
from rich.console import Console
from dataclasses import dataclass
import tyro
from tqdm import tqdm
import numpy as np

from render_ortho import OrthoRender, eval_setup, save_img_u8, save_img_f32, save_vis_depth
from utils import get_tile_configs

CONSOLE = Console(width=120)

@dataclass
class OrthoSplitRender(OrthoRender):
    # Path to config YAML file.
    # load_config: Optional[Path] = None
    load_config: Path = Path("output/songya100_split/scaffold-gs/2025-10-10_214537/config.yml")

    def calculate_gsd_(self, tile_configs) -> tuple:
        list_gsd = []
        list_tile_size = []
        list_camera_height = []
        for config in tile_configs:
            _, scene, _ = eval_setup(config_path=config)
            gsd, tile_size, camera_height = self.calculate_gsd(scene)
            list_gsd.append(gsd)
            list_tile_size.append(tile_size)
            list_camera_height.append(camera_height)
        if self.gsd is None:
            self.gsd = max(list_gsd)
        if self.tile_size is None:
            self.tile_size = mean(list_tile_size)
        if self.camera_height is None:
            self.camera_height = mean(list_camera_height)
        return self.gsd, self.tile_size, self.camera_height

    def main_(self) -> None:
        config, tile_configs = get_tile_configs(self.load_config)

        output_dir = os.path.join(config.get_base_dir(), 'ortho')
        os.makedirs(output_dir, exist_ok=True)

        # estimate gsd, tile-size, camera-height
        gsd, tile_size, camera_height = self.calculate_gsd_(tile_configs)
        CONSOLE.print(f"Ground Sample Distance (m): {gsd}")
        CONSOLE.print(f"Tile Size (m): {tile_size}")
        CONSOLE.print(f"Camera Height (m): {camera_height}")

        for i in tqdm(range(len(tile_configs)), desc="render ortho for each tile"):
            cameras, list_render, list_depth, (scale, tx, ty, tz) = self.main(tile_configs[i], save=False)

            tile_name = os.fspath(config.partitioner.config_of_tiles[i])
            out_dir = os.path.join(config.get_base_dir(), 'ortho', tile_name)
            render_path = os.path.join(out_dir, "renders")
            vis_path = os.path.join(out_dir, "depths")
            coordinate_path = os.path.join(out_dir, "coordinate")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(vis_path, exist_ok=True)
            os.makedirs(coordinate_path, exist_ok=True)

            for idx, viewpoint_cam in tqdm(enumerate(cameras), desc="save render and depth"):
                save_img_u8(list_render[idx], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

                if len(list_depth) > 0:
                    depth = (self.camera_height - list_depth[idx]) * scale + tz
                    save_img_f32(depth, os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
                    
                    distance_limits = np.percentile(depth.flatten(), [3, 100 - 3])
                    lo, hi = [x for x in distance_limits]
                    # lo, hi = [np.log(x-lo) for x in distance_limits]
                    save_vis_depth(depth, lo, hi, os.path.join(vis_path, 'depth_vis_{0:05d}'.format(idx) + ".png"))

                with open(os.path.join(coordinate_path, '{0:05d}'.format(idx) + ".txt"), 'w') as f:
                    f.write(f"{viewpoint_cam.image_width}\n")
                    f.write(f"{viewpoint_cam.image_height}\n")
                    f.write(f"{self.gsd}\n")
                    f.write(f"{viewpoint_cam.bbx[0] * scale + tx}\n")
                    f.write(f"{viewpoint_cam.bbx[2] * scale + ty}\n")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(OrthoSplitRender).main_()

if __name__ == "__main__":
    entrypoint()
