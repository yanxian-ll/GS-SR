from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List

from gssr.dataloader.colmap_dataloader import ColmapDataLoaderConfig
from gssr.scene.base_scene import Scene, SceneConfig
from gssr.gaussian.vanilla_gaussian import VanillaGaussianConfig
from gssr.utils.image_utils import ssim

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

@dataclass
class VanillaSceneConfig(SceneConfig):
    _target: type = field(default_factory=lambda: VanillaScene)

    dataloader: ColmapDataLoaderConfig = ColmapDataLoaderConfig()
    gaussians: VanillaGaussianConfig = VanillaGaussianConfig()
    lambda_dssim: float = 0.2
    antialiasing: bool = False


class VanillaScene(Scene):
    config: VanillaSceneConfig

    def l1_loss(self, network_output, gt):
        return torch.abs((network_output - gt)).mean()

    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = outputs["render"].to(self.device)
        gt_image = viewpoint_cam.original_image.to(self.device)
        loss_dict['L1_loss'] = (1.0 - self.config.lambda_dssim) * self.l1_loss(image, gt_image)
        loss_dict['ssim_loss'] = self.config.lambda_dssim * (1.0 - ssim(image, gt_image))        
        return loss_dict
    
    def generate_gaussians(self, viewpoint_camera):
        means3D = self._gaussians.get_xyz
        opacity = self._gaussians.get_opacity

        # # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # # scaling / rotation by the rasterizer.
        # scales = None
        # rotations = None
        # cov3D_precomp = None
        # # if self.config.compute_cov3D_python:
        # if True:
        #     cov3D_precomp = self._gaussians.get_covariance(self.config.scaling_modifier)
        # else:
        #     scales = self._gaussians.get_scaling
        #     rotations = self._gaussians.get_rotation
        scales = self._gaussians.get_scaling
        rotations = self._gaussians.get_rotation
        cov3D_precomp = None

        # # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        # shs = None
        # colors_precomp = None
        # override_color = None
        # if override_color is None:
        #     if self.config.convert_SHs_python:
        #         shs_view = self._gaussians.get_features.transpose(1, 2).view(-1, 3, (self._gaussians.max_sh_degree+1)**2)
        #         dir_pp = (self._gaussians.get_xyz - viewpoint_camera.camera_center.repeat(self._gaussians.get_features.shape[0], 1))
        #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        #         sh2rgb = eval_sh(self._gaussians.active_sh_degree, shs_view, dir_pp_normalized)
        #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        #     else:
        #         shs = self._gaussians.get_features
        # else:
        #     colors_precomp = override_color
        
        shs = self._gaussians.get_features
        colors_precomp = None
        other_output = {}  # used for scaffold
        return means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output

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
            debug=self.config.debug,
            antialiasing=self.config.antialiasing
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": 1.0 / (depth_image)}
    
    @torch.no_grad()
    def render_ortho(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0

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
            debug=self.config.debug,
            antialiasing=self.config.antialiasing,
            ortho_rendering=True,
            dx=viewpoint_camera.ground_width,
            dy=viewpoint_camera.ground_height,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": 1.0 / depth_image}

    @torch.no_grad()
    def simp_render(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):

        from simp_gaussian_rasterization import GaussianRasterizationSettings as simp_GaussianRasterizationSettings
        from simp_gaussian_rasterization import GaussianRasterizer as simp_GaussianRasterizer

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = simp_GaussianRasterizationSettings(
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
            debug=self.config.debug,
            antialiasing=self.config.antialiasing
        )

        rasterizer = simp_GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, weights, counts, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "weights": weights,
                "counts": counts,
                "depth": 1.0 / (depth_image)}
    
    @torch.no_grad()
    def simp_render_ortho(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):

        from simp_ortho_gaussian_rasterization import GaussianRasterizationSettings as simp_OrthoGaussianRasterizationSettings
        from simp_ortho_gaussian_rasterization import GaussianRasterizer as simp_OrthoGaussianRasterizer

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = simp_OrthoGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,

            dx=viewpoint_camera.ground_width,
            dy=viewpoint_camera.ground_height,

            bg=self.background,
            scale_modifier=self.config.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self._gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=self.config.debug,
            antialiasing=self.config.antialiasing
        )

        rasterizer = simp_OrthoGaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, weights, counts, depth_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "weights": weights,
                "counts": counts,
                "depth": 1.0 / (depth_image)}
    

