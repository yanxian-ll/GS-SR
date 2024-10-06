# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base Configs"""
from __future__ import annotations

import yaml
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from rich.console import Console

warnings.filterwarnings("ignore", module="torchvision")

CONSOLE = Console(width=120)

# Pretty printing class
class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)

# Machine related configs
@dataclass
class MachineConfig(PrintableConfig):
    """Configuration of machine setup"""

    seed: int = 42
    """random seed initilization"""
    num_gpus: int = 1
    """total number of gpus available for train/eval"""
    num_machines: int = 1
    """total number of distributed machines available (for DDP)"""
    machine_rank: int = 0
    """current machine's rank (for DDP)"""
    dist_url: str = "auto"
    """distributed connection point (for DDP)"""


@dataclass
class TrainerConfig(PrintableConfig):
    iterations: int = 30_000
    test_iterations: List[int] = field(default_factory=lambda:[30_000])
    save_iterations: List[int] = field(default_factory=lambda:[30_000])
    relative_gaussian_dir: Path = Path("point_cloud/")

    checkpoint_iterations: List[int] = field(default_factory=list)
    relative_ckpt_dir: Path = Path("chkpnt/")
    save_only_latest_checkpoint: bool = False

    # optional parameters if we want to resume training
    load_ckpt_dir: Optional[Path] = None
    load_ckpt_step: Optional[int] = None
    load_gaussian_dir: Optional[Path] = None
    load_gaussian_step: Optional[int] = None
    load_config: Optional[Path] = None

@dataclass
class PartitionConfig(PrintableConfig):
    need_partition: bool = True
    num_col: int = 4
    num_row: int = 1
    extend_ratio: float = 0.1
    visibility_threshold: float = 0.5
    config_of_tiles: List[Path] = field(default_factory=list)


from gssr.scene.base_scene import SceneConfig

@dataclass
class Config(PrintableConfig):
    """Full config contents"""

    source_path: Optional[str] = None
    output_path: str = "./output"
    method_name: Optional[str] = None # required
    experiment_name: Optional[str] = None
    timestamp: str = "{timestamp}"
    eval: bool = False

    machine: MachineConfig = MachineConfig()
    trainer: TrainerConfig = TrainerConfig()
    scene: SceneConfig = SceneConfig()
    partitioner: PartitionConfig = PartitionConfig()

    writer: str = "tensorboard"
    relative_log_dir: Path = Path("logs")
    relative_config_dir: Path = Path("./")

    def is_tensorboard_enabled(self) -> bool:
        """Checks if tensorboard is enabled."""
        return "tensorboard" == self.writer
    
    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            self.experiment_name = str(self.source_path.split('/')[-1])

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config"
        self.set_experiment_name()
        return Path(f"{self.output_path}/{self.experiment_name}/{self.method_name}/{self.timestamp}")
    
    def get_gaussian_dir(self) -> Path:
        """Retrieve the point cloud directory"""
        return Path(self.get_base_dir() / self.trainer.relative_gaussian_dir)

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.trainer.relative_ckpt_dir)
    
    def get_config_dir(self) -> Path:
        return Path(self.get_base_dir() / self.relative_config_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        config_dir = self.get_config_dir()
        assert config_dir is not None
        config_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = config_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")

