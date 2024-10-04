from dataclasses import dataclass, field
from typing import Any, Dict, List
from torch._tensor import Tensor
from gssr.scene.scaffold_scene import ScaffoldScene, ScaffoldSceneConfig
from gssr.scene.pgsr_scene import PGSRSceneConfig, PGSRScene

@dataclass
class ScaffoldPGSRSceneConfig(PGSRSceneConfig, ScaffoldSceneConfig):
    _target: type = field(default_factory=lambda: ScaffoldPGSRScene)

class ScaffoldPGSRScene(PGSRScene, ScaffoldScene):
    """PGSRSScene didn't define generate_gaussians, so will use ScafflodScene-generate_gaussians"""
    config: ScaffoldPGSRSceneConfig

    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        """use PGSRScene-get_loss_dict first"""
        # pgsr-loss
        loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        # scaffold-loss
        loss_dict["scaling_loss"] = self.config.lambda_scaling * outputs["scaling"].prod(dim=1).mean()
        return loss_dict
