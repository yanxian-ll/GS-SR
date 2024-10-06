from pathlib import Path
from dataclasses import dataclass
import tyro

from gssr.utils.vastgaussian_utils import split_scene
from typing import Optional


@dataclass
class SceneSpliter:
    """partition the scene (only support colmap form)"""
    source_path: Path
    output_path: Optional[Path] = None
    num_col: Optional[int] = 4
    num_row: Optional[int] = 1
    extend_ratio: float = 0.1
    visibility_threshold: float = 0.5

    def main(self) -> None:
        if self.output_path is None:
            self.output_path = self.source_path

        split_scene(self.source_path, self.output_path, 
                    self.num_col, self.num_row, 
                    transform_matrix=None, 
                    ratio=self.extend_ratio, 
                    threshold=self.visibility_threshold, 
                    input_format="", output_format=".txt")


def entrypoint():
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(SceneSpliter).main()


if __name__ == "__main__":
    entrypoint()

