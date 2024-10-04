"""
Abstracts for the Pipeline class.
"""
from __future__ import annotations

import os
from torch import nn
from torch.nn import Parameter
from dataclasses import dataclass, field
from typing import Dict, List

from gssr.configs import base_config as cfg
from gssr.engine.callbacks import TrainingCallback
from gssr.pointcloud import BasicPointCloud


@dataclass
class GaussianModelConfig(cfg.InstantiateConfig):
    _target: type = field(default_factory=lambda: GaussianModel)
    max_sh_degree: int = 3
    percent_dense: float = 0.01
    sampling_ratio: int = 1
    """sampling the input point cloud"""

class GaussianModel:
    config: GaussianModelConfig
    def __init__(
        self,
        config: GaussianModelConfig,
        device: str = 'cuda',
        world_size: int = 1,
        local_rank: int = 0,
    ) -> None:
        self.config = config
        self.device =  device
        self.world_size = world_size
        self.local_rank = local_rank

        self.callbacks = None
        self.optimizer = None
        self.spatial_lr_scale = 0  # equal to camera-extent
        self.white_background = False

        self.max_sh_degree = self.config.max_sh_degree
        self.active_sh_degree = 0
        self.percent_dense = self.config.percent_dense

    
    def create_from_data(self, pcd : BasicPointCloud, cameras : Dict, spatial_lr_scale : float):
        """Create gaussians from data"""

    def get_training_callbacks(self) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        return []
    
    def setup_optimizers(self):
        """Setup optimizers"""

    def capture(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the gaussians."""

    def restore(self):
        """"Restore"""

    def save_gaussians(self, path: str):
        """Save gaussians(PLY, MLp, ...) to the given path"""
    
    def load_gaussians(self, path: str):
        """Load gaussians from the given path"""
    
    def save_mlp_checkpoints(self, path):
        """for scaffold"""
    
    def load_mlp_checkpoints(self, path):
        """for scaffold"""

    def densify(self, step, **kwags):
        """densify gaussians"""
        