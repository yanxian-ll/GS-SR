from dataclasses import dataclass, field
from typing import Any, Dict, List
from torch._tensor import Tensor
from gssr.scene.octree_scene import OctreeScene, OctreeSceneConfig
from gssr.scene.twodgs_scene import TwoDGSScene, TwoDGSSceneConfig

@dataclass
class Octree2DGSSceneConfig(TwoDGSSceneConfig, OctreeSceneConfig):
    _target: type = field(default_factory=lambda: Octree2DGSScene)

class Octree2DGSScene(TwoDGSScene, OctreeScene):
    config: Octree2DGSSceneConfig
    
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
