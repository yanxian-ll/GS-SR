from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from torch._tensor import Tensor
import math
from tqdm import tqdm

from gssr.utils.point_utils import depth_to_normal
from gssr.scene.vanilla_scene import VanillaScene, VanillaSceneConfig

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@dataclass
class TwoDGSSceneConfig(VanillaSceneConfig):
    _target: type = field(default_factory=lambda: TwoDGSScene)
    lambda_dist: float = 0.0
    lambda_normal: float = 0.05
    depth_ratio: float = 0.0
    """depth_ratio = 0 or 1"""

class TwoDGSScene(VanillaScene):
    config: TwoDGSSceneConfig

    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        # regularization
        lambda_normal = self.config.lambda_normal if step > 7000 else 0.0
        lambda_dist = self.config.lambda_dist if step > 3000 else 0.0

        rend_dist, rend_normal, surf_normal = outputs["rend_dist"], outputs['normal'], outputs['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        loss_dict["normal_loss"] = lambda_normal * (normal_error).mean()
        loss_dict["dist_loss"] = lambda_dist * (rend_dist).mean()
        return loss_dict
    
    def render(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=self.config.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self._gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.config.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points
        
        rendered_image, radii, allmap = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        rets =  {"render": rendered_image,
                "viewspace_points": means2D,
                "visibility_filter" : radii > 0,
                "radii": radii,
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
        
        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
        
        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1; 
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1-self.config.depth_ratio) + (self.config.depth_ratio) * render_depth_median
        
        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2,0,1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

        rets.update({
                'rend_alpha': render_alpha,
                'rend_dist': render_dist,
                'surf_normal': surf_normal,
                'depth': surf_depth,
                'normal': render_normal,
        })

        return rets
