from __future__ import annotations

import torch
import os
from torch import nn
from torch.nn import Parameter
from dataclasses import dataclass, field
from typing import Any, Dict, List
from pathlib import Path
from tqdm import tqdm

from gssr.configs import base_config as cfg
from gssr.engine.callbacks import TrainingCallback
from gssr.gaussian.base_gaussian import GaussianModel
from gssr.dataloader.colmap_dataloader import DataLoader


@dataclass
class SceneConfig(cfg.InstantiateConfig):
    _target: type = field(default_factory=lambda: Scene)

    convert_SHs_python: bool = False
    compute_cov3D_python: bool = False
    debug: bool = False

    random_background: bool = False
    scaling_modifier: float = 1.0

    relative_gaussian_dir: str = "point_cloud"


class Scene(nn.Module):
    dataloader: DataLoader
    _gaussians: GaussianModel

    def __init__(
        self, 
        config: SceneConfig,
        source_dir: str,
        eval: bool = False,
        device: str = 'cuda',
        world_size: int = 1,
        local_rank: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.source_dir = source_dir
        self.evaluation = eval
        self.world_size = world_size
        self.local_rank = local_rank

        ## Setup Dataloader
        self.dataloader = config.dataloader.setup(
            source_dir=source_dir, eval=eval,
            world_size=world_size, local_rank=local_rank
        )
        assert self.dataloader.getTrainData() is not None, "Missing input dataset"
        self.cameras_extent = self.dataloader.cameras_extent
        self.background = self.dataloader.background
        self.white_background = self.dataloader.config.white_background

        ## Setup Gaussians and Optimizers
        self._gaussians = config.gaussians.setup(world_size=world_size, local_rank=local_rank)
        self._gaussians.create_from_data(self.dataloader.point_cloud, self.dataloader.train_dataset, self.cameras_extent)
        
        ## Pre-Calculate background
        self.background = self.get_background()
    
    def forward(self):
        """Blank forward method

        This is an nn.Module, and so requires a forward() method normally, although in our case
        we do not need a forward() method"""
        raise NotImplementedError

    def get_training_callbacks(self) -> List[TrainingCallback]:
        """Returns the training callbacks from both the Dataloader and the Model."""
        dataloader_callbacks = self.dataloader.get_training_callbacks()
        model_callbacks = self._gaussians.get_training_callbacks()
        callbacks = dataloader_callbacks + model_callbacks
        return callbacks

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Get the param groups for the pipeline."""

    def get_background(self):
        bg = torch.rand((3), device=self.device) if self.config.random_background \
            else torch.tensor(self.background, dtype=torch.float32, device=self.device)
        return bg
    
    def generate_gaussians(self, viewpoint_camera):
        """"""
    
    def render(self, viewpoint_camera) -> Dict[str, torch.Tensor]:
        """Rendering"""
    
    def get_metrics_dict(self, outputs, viewpoint_cam) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics."""
    
    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict."""
    
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict."""
        viewpoint_cam = self.dataloader.next_train()
        
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)

        metrics_dict = self.get_metrics_dict(model_outputs, viewpoint_cam)
        loss_dict = self.get_loss_dict(model_outputs, viewpoint_cam, step, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
    
    def densify(self, step, output):
        self._gaussians.densify(step, **output)
    
    @torch.no_grad()
    def eval_render(self, viewpoint_cam):
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)
        return model_outputs
    