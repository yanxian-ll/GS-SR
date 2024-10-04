from __future__ import annotations

from random import randint
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Type
from tqdm import tqdm

from gssr.configs import base_config as cfg
from gssr.engine.callbacks import TrainingCallback


@dataclass
class DataLoaderConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: DataLoader)
    shuffle: bool = True
    llffhold: int = 8
    resolution_scales: List[float] = field(default_factory=lambda: [1.0])
    images: str = 'images'
    device: str = 'cuda'
    resolution: int = -1
    white_background: bool = False

class DataLoader(nn.Module):
    config: DataLoaderConfig
    def __init__(
        self,
        config: DataLoaderConfig,
        source_dir: str,
        eval: bool = False,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__()
        self.config = config
        self.source_dir = source_dir
        self.evaluation = eval
        self.world_size = world_size
        self.local_rank = local_rank

        self.train_dataset = {}
        self.test_dataset = {}
        self.cameras_extent = 0.0
        self.point_cloud = None

        self.viewpoint_stack = None
        self.background = [1,1,1] if self.config.white_background else [0,0,0]

        for resolution_scale in self.config.resolution_scales:
            print("Loading Training Data")
            self.train_dataset[resolution_scale] = None
            print("Loading Test Data")
            self.test_dataset[resolution_scale] = None

    def getTrainData(self, scale=1.0):
        return self.train_dataset[scale]
    
    def getTestData(self, scale=1.0):
        return self.test_dataset[scale]
    
    def next_train(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.getTrainData().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
    
    def get_training_callbacks(self) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        return []
    