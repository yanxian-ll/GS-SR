from dataclasses import dataclass, field
from typing import Any, Dict, List
from torch._tensor import Tensor
import random

from gssr.scene.octree_scene import OctreeScene, OctreeSceneConfig
from gssr.scene.pgsr_scene import PGSRSceneConfig, PGSRScene

@dataclass
class OctreePGSRSceneConfig(PGSRSceneConfig, OctreeSceneConfig):
    _target: type = field(default_factory=lambda: OctreePGSRScene)

class OctreePGSRScene(PGSRScene, OctreeScene):
    """PGSRSScene didn't define generate_gaussians, so will use ScafflodScene-generate_gaussians"""

    config: OctreePGSRSceneConfig
    
    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        """use PGSRScene-get_loss_dict first"""
        # pgsr-loss
        loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        # scaffold-loss
        loss_dict["scaling_loss"] = self.config.lambda_scaling * outputs["scaling"].prod(dim=1).mean()
        return loss_dict
    
    def get_train_loss_dict(self, step: int):
        viewpoint_cam = self.dataloader.next_train()

        self._gaussians.set_anchor_mask(viewpoint_cam.camera_center, step, viewpoint_cam.resolution_scale)
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)
        metrics_dict = self.get_metrics_dict(model_outputs, viewpoint_cam)

        ## render near camera
        near_model_outputs = None
        near_cam = None
        if step > 7000:
            near_cam = None if len(viewpoint_cam.near_ids) == 0 else self.dataloader.getTrainData()[random.sample(viewpoint_cam.near_ids, 1)[0]]
            self._gaussians.set_anchor_mask(near_cam.camera_center, step, near_cam.resolution_scale)
            means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(near_cam)
            near_model_outputs = self.render(near_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
            near_model_outputs.update(other_output)

        loss_dict = self.get_loss_dict(model_outputs, viewpoint_cam, step, metrics_dict, near_cam=near_cam, nearest_render_pkg=near_model_outputs)
        return model_outputs, loss_dict, metrics_dict
