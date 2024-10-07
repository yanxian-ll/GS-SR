from pathlib import Path
from dataclasses import dataclass
import tyro
from rich.console import Console
from gssr.utils.vastgaussian_utils import *
from typing import Optional
import shutil
from tqdm import tqdm

CONSOLE = Console(width=120)

@dataclass
class SceneSpliter:
    """partition the scene (only support colmap form)"""
    source_path: Path
    output_path: Optional[Path] = None
    num_col: Optional[int] = None
    num_row: Optional[int] = None
    max_num_images: Optional[int] = 200
    extend_ratio: float = 0.1
    visibility_threshold: float = 0.5
    transform_file: Optional[str] = None

    def main(self) -> None:
        if self.output_path is None:
            self.output_path = self.source_path
        
        os.makedirs(self.output_path, exist_ok=True)

        ## transform coordinate, so that the Z-axis of the world coordinate is perpendicular to the ground plane
        if self.transform_file is not None:
            cameras, images, points3D = transform_colmap(
                os.path.join(self.source_path, "sparse/0"), 
                os.path.join(self.output_path, "sparse/aligned"), 
                self.transform_file)
        else:
            cameras, images, points3D = read_model(path=os.path.join(self.source_path, "sparse/0"))

        # Camera-position-based region division
        CONSOLE.log("camera-position-based region division")
        tiles = camera_position_based_region_division(images, self.num_col, self.num_row, self.max_num_images)
        
        # Position-based data selection
        CONSOLE.log("position-based data selection")
        tiles = position_based_data_selection(tiles, images, points3D, ratio=self.extend_ratio)
        
        # Visibility-based camera selection
        CONSOLE.log("visibility-based camera selection")
        tiles = visibility_based_camera_selection(tiles, images, cameras, threshod=self.visibility_threshold)

        # Coverage-based point selection
        CONSOLE.log("coverage-based point selection")
        tiles = coverage_based_point_selection(tiles, points3D)

        # write tiles as colmap form
        for i, tile in tqdm(enumerate(tiles)):
            images_tile = tile['images']
            points3D_tile = tile['points3D']
            images_tile_dict, points3D_tile_dict = {}, {}
            for img in images_tile: images_tile_dict[img.id]=img
            for point in points3D_tile: points3D_tile_dict[point.id]=point
            # write sparse/0/...
            tile_name = 'tile_%04d' % i
            output_tile = os.path.join(self.output_path, f"{tile_name}/sparse/0")
            os.makedirs(output_tile, exist_ok=True)
            write_model(cameras, images_tile_dict, points3D_tile_dict, path=output_tile, ext=".txt")
            CONSOLE.log(f"write {tile_name}, num-images: {len(images_tile)}, num-points: {len(points3D_tile)}")

            # write box for merge
            box = tile['box']
            with open(os.path.join(self.output_path, f"{tile_name}/box.txt"), 'w') as f:
                f.write(f"mx Mx my My\n")
                f.write(f"{box[0]} {box[1]} {box[2]} {box[3]}")

            # copy images
            image_path = os.path.join(self.output_path, f"{tile_name}/images")
            if os.path.exists(image_path): shutil.rmtree(image_path)
            os.makedirs(image_path, exist_ok=True)
            for img in images_tile:
                orig_path = os.path.join(self.source_path, 'images', img.name)
                new_path = os.path.join(image_path, img.name)
                shutil.copy(orig_path, new_path)
            

def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(SceneSpliter).main()


if __name__ == "__main__":
    entrypoint()

