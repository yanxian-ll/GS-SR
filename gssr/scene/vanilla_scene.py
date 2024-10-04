from __future__ import annotations

import math
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List

from gssr.dataloader.colmap_dataloader import ColmapDataLoaderConfig
from gssr.scene.base_scene import Scene, SceneConfig
from gssr.gaussian.vanilla_gaussian import VanillaGaussianConfig

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


@dataclass
class VanillaSceneConfig(SceneConfig):
    _target: type = field(default_factory=lambda: VanillaScene)

    dataloader: ColmapDataLoaderConfig = ColmapDataLoaderConfig()
    gaussians: VanillaGaussianConfig = VanillaGaussianConfig()
    lambda_dssim: float = 0.2


class VanillaScene(Scene):
    config: VanillaSceneConfig

    def l1_loss(self, network_output, gt):
        return torch.abs((network_output - gt)).mean()
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    def _gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
        return gauss / gauss.sum()
    
    def ssim(self, img1, img2, window_size=11, size_average=True):
        channel = img1.size(-3)
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        if img1.is_cuda:
            window = window.cuda(img1.get_device())
        window = window.type_as(img1)
        return self._ssim(img1, img2, window, window_size, channel, size_average)

    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        image = outputs["render"].to(self.device)
        gt_image = viewpoint_cam.original_image.to(self.device)
        loss_dict['L1_loss'] = (1.0 - self.config.lambda_dssim) * self.l1_loss(image, gt_image)
        loss_dict['ssim_loss'] = self.config.lambda_dssim * (1.0 - self.ssim(image, gt_image))
        return loss_dict
    
    def generate_gaussians(self, viewpoint_camera):
        means3D = self._gaussians.get_xyz
        opacity = self._gaussians.get_opacity

        # # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # # scaling / rotation by the rasterizer.
        # scales = None
        # rotations = None
        # cov3D_precomp = None
        # if self.config.compute_cov3D_python:
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
        other_output = {}
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
            debug=self.config.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii = rasterizer(
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
                "radii": radii}
