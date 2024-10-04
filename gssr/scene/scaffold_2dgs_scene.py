from dataclasses import dataclass, field
from typing import Any, Dict, List
from torch._tensor import Tensor
from gssr.scene.scaffold_scene import ScaffoldScene, ScaffoldSceneConfig
from gssr.scene.twodgs_scene import TwoDGSScene, TwoDGSSceneConfig

@dataclass
class Scaffold2DGSSceneConfig(TwoDGSSceneConfig, ScaffoldSceneConfig):
    _target: type = field(default_factory=lambda: Scaffold2DGSScene)

class Scaffold2DGSScene(TwoDGSScene, ScaffoldScene):
    config: Scaffold2DGSSceneConfig
    
    def generate_gaussians(self, viewpoint_camera):
        xyz, opacity, scaling, rot, cov3D_precomp, shs, color, other_output =  super().generate_gaussians(viewpoint_camera)
        # only keep first-two scales
        scaling = scaling[:, :2]
        other_output["scaling"] = other_output["scaling"][:, :2]
        return xyz, opacity, scaling, rot, cov3D_precomp, shs, color, other_output
    
    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        # 2dgs-loss
        loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        # scaffold-loss
        loss_dict["scaling_loss"] = self.config.lambda_scaling * outputs["scaling"].prod(dim=1).mean()
        return loss_dict
