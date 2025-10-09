from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from torch._tensor import Tensor
import math
from einops import repeat

from gssr.scene.base_scene import SceneConfig
from gssr.utils.point_utils import depth_to_normal
from gssr.scene.vanilla_scene import VanillaScene, VanillaSceneConfig
from gssr.utils.image_utils import ssim

@dataclass
class SatelliteScaffoldSceneConfig(VanillaSceneConfig):
    _target: type = field(default_factory=lambda: SatelliteScaffoldScene)
    lambda_scaling: float = 0.01

class SatelliteScaffoldScene(VanillaScene):
    config: SatelliteScaffoldSceneConfig

    def __init__(self, config: SceneConfig, source_dir: str, eval: bool = False, device: str = 'cuda', world_size: int = 1, local_rank: int = 0) -> None:
        super().__init__(config, source_dir, eval, device, world_size, local_rank)
        self._gaussians.set_appearance(len(self.dataloader.getTrainData()))  # set appearance first
        self._gaussians.train()
    

    def render(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):
        from satellite_diff_gaussian_rasterization import GaussianRasterizationSettings as SatelliteGaussianRasterizationSettings
        from satellite_diff_gaussian_rasterization import GaussianRasterizer as SatelliteGaussianRasterizer

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = SatelliteGaussianRasterizationSettings(
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

        rasterizer = SatelliteGaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        color = colors_precomp[:, :8]
        weights = colors_precomp[:, 7:8]

        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        rendered_image, radii, depth_image, weights_image = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = color,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            weights = weights,
            cov3D_precomp = cov3D_precomp)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {"render": rendered_image[:3, :, :],
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "depth": 1.0 / (depth_image),

                'shadow': weights_image,
                "transient": rendered_image[3:6, :, :],
                "uncertainty": rendered_image[6:7, :, :],
                "shadow2": rendered_image[7:8, :, :]}
    
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
    

    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, is_training=False):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self._gaussians.get_anchor.shape[0], dtype=torch.bool, device = self._gaussians.get_anchor.device)
        
        base_feat = self._gaussians._anchor_feat[visible_mask]
        anchor = self._gaussians.get_anchor[visible_mask]
        grid_offsets = self._gaussians._offset[visible_mask]
        grid_scaling = self._gaussians.get_scaling[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        # TODO
        anchor_encoding = self._gaussians.get_mapping_xyz(anchor)
        view_encoding = self._gaussians.get_mapping_view(ob_view)

        feat = torch.cat([base_feat, anchor_encoding], dim=1)
        feat_encoding = self._gaussians.get_featurebank_mlp(feat)

        # get offset's opacity
        neural_opacity = self._gaussians.get_opacity_mlp(torch.cat([feat_encoding, view_encoding], dim=1))
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)
        opacity = neural_opacity[mask]

        # get offset's albedo
        color = self._gaussians.get_color_mlp(feat_encoding)
        color = color.reshape([anchor.shape[0]*self._gaussians.n_offsets, 3])# [mask]

        # get offset's cov
        scale_rot = self._gaussians.get_cov_mlp(feat_encoding)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self._gaussians.n_offsets, 7]) # [mask]
        
        if self._gaussians.appearance_dim > 0:
            camera_indicies = torch.ones_like(feat_encoding[:,0], dtype=torch.long, device=feat_encoding.device) * viewpoint_camera.uid
            appearance = self._gaussians.get_appearance(camera_indicies)
            transient_encoding = self._gaussians.get_transient_encoding_mlp(torch.cat([feat_encoding, appearance], dim=1))
        else:
            transient_encoding = self._gaussians.get_transient_encoding_mlp(feat_encoding)

        # get offset's uncertainty and transient
        transient = self._gaussians.get_transient_mlp(transient_encoding)
        uncertainty = self._gaussians.get_uncertainty_mlp(transient_encoding)

        transient = transient.reshape([anchor.shape[0]*self._gaussians.n_offsets, 3])[mask]
        uncertainty = uncertainty.reshape([anchor.shape[0]*self._gaussians.n_offsets, 1])[mask]

        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self._gaussians.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
        rot = self._gaussians.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets

        color = torch.concatenate([color, transient, uncertainty], dim=-1)

        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot
    
    def prefilter_voxel(self, viewpoint_camera):
        """Render the scene. """
        from scaffold_filter import GaussianRasterizationSettings, GaussianRasterizer

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

        means3D = self._gaussians.get_anchor
        cov3D_precomp = None
        scales = self._gaussians.get_scaling
        rotations = self._gaussians.get_rotation

        radii_pure = rasterizer.visible_filter(
            means3D = means3D,
            scales = scales[:,:3],
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        return radii_pure > 0
    
    def generate_gaussians(self, viewpoint_camera):
        # prefilter
        voxel_visible_mask = self.prefilter_voxel(viewpoint_camera)

        is_training = self._gaussians.get_color_mlp.training

        # get sky color
        sun_view = viewpoint_camera.sun_camera.sun_view.unsqueeze(0)
        view_encoding = self._gaussians.get_mapping_view(sun_view)
        sky_color = self._gaussians.get_sky_color_mlp(view_encoding)
        
        if is_training:
            xyz, color, opacity, scaling, rot, neural_opacity, mask = self.generate_neural_gaussians(viewpoint_camera, voxel_visible_mask, is_training=is_training)
            other_output = {
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "opacity": opacity,
                "voxel_visible_mask": voxel_visible_mask,
                "sky_color": sky_color,
            }
        else:
            xyz, color, opacity, scaling, rot = self.generate_neural_gaussians(viewpoint_camera, voxel_visible_mask, is_training=is_training)
            other_output = {
                "scaling": scaling,
                "voxel_visible_mask": voxel_visible_mask,
                "sky_color": sky_color,
            }
        
        cov3D_precomp = None
        shs = None
        return xyz, opacity, scaling, rot, cov3D_precomp, shs, color, other_output
    

    def uncertainty_aware_loss(self, network_output, gt, uncertainty, min_beta=0.05):
        beta = uncertainty + min_beta
        color_loss = ((network_output - gt) ** 2 / (2 * beta ** 2)).mean()
        logbeta = (3 + torch.log(beta).mean()) / 2
        return color_loss, logbeta


    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict."""
        viewpoint_cam = self.dataloader.next_train()
        
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)

        with torch.no_grad():
            simp_outputs = self.simp_render_ortho(viewpoint_cam.sun_camera, means3D.detach(), opacity.detach(), scales.detach(), rotations.detach(), cov3D_precomp, shs, colors_precomp[:,:3].detach())
        
            weights = simp_outputs['weights'].reshape(-1,1)
            counts = simp_outputs['counts'].reshape(-1,1)
            weights[counts <= 0] = 0.0
            weights[counts > 0] = weights[counts > 0] / counts[counts > 0]
            weights = torch.nn.functional.softmax(weights)
            q = torch.quantile(weights, q=0.50)
            weights = weights - q
            weights[weights<=0] = (weights[weights<=0] / (q + 1e-10)) * 5.0
            weights[weights>0] = (weights[weights>0] / (weights.max()+1e-10)) * 5.0
            weights = torch.sigmoid(weights)
            # weights = torch.clip(weights, 0, 1)
            # q1 = torch.quantile(weights, q=0.50)
            # q2 = torch.quantile(weights, q=0.95)
            # weights[weights>q1] = 1.0
            # weights = weights / (weights.max()+1e-10)

        # render
        colors_precomp = torch.concatenate([colors_precomp, weights], dim=-1)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)

        metrics_dict = self.get_metrics_dict(model_outputs, viewpoint_cam)
        loss_dict = self.get_loss_dict(model_outputs, viewpoint_cam, step, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
    
    
    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        # loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        loss_dict = {}
        albedo = outputs["render"].to(self.device)
        transient = outputs["transient"].to(self.device)
        uncertainty = outputs["uncertainty"].to(self.device)
        shadow = outputs["shadow2"].to(self.device)
        sky = outputs['sky_color'].to(self.device)

        # if step < 7000:
        #     s = shadow.repeat(3,1,1)
        #     sky = torch.tensor([0.0, 0.0, 0.0]).cuda()
        #     irradiance = s + (1-s) * sky.reshape(3,1,1).repeat(1,s.shape[1],s.shape[2])
        # else:
        
        s = shadow * transient
        irradiance = s + (1-s) * sky.reshape(3,1,1).repeat(1,s.shape[1],s.shape[2])

        # affine = self._gaussians.get_color_affine[viewpoint_cam.uid, :]
        # image = affine[0] * albedo * irradiance + affine[1]
        image = albedo * irradiance
        # image = albedo

        gt_image = viewpoint_cam.original_image.to(self.device)
        # loss_dict['L1_loss'] = (1.0 - self.config.lambda_dssim) * self.l1_loss(image, gt_image)
        color_loss, logbeta = self.uncertainty_aware_loss(image, gt_image, uncertainty)
        loss_dict['L1_loss'] = (1.0 - self.config.lambda_dssim) * color_loss
        loss_dict['logbeta'] = (1.0 - self.config.lambda_dssim) * logbeta
        loss_dict['ssim_loss'] = self.config.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_dict['scaling_loss'] = self.config.lambda_scaling * outputs["scaling"].prod(dim=1).mean()

        # if step < 7_000:
        #     loss_dict['sky_loss'] = 0.5 * (sky - 1.0).abs().mean()

        # if step < 4000 and step > 3000:
            # opacity = outputs['opacity']
            # loss_dict['entropy_loss'] = 0.001 * (-opacity * torch.log(opacity+1e-10) - (1-opacity)*torch.log(1-opacity + 1e-10)).mean()
                
        # 可视化
        with torch.no_grad():
            if step % 100 == 1:
            # if True:
                import cv2
                import numpy as np
                import os

                gt_image = viewpoint_cam.original_image.to(self.device)
                gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                img_show = (image.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

                depth = outputs['depth'].squeeze().detach().cpu().numpy()
                depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                depth_show = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

                shadow = (shadow.squeeze().detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                # shadow_show = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)
                shadow_show = np.stack([shadow, shadow, shadow], axis=-1)

                transient_show = (transient.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                s_show = (s.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

                # s = (s.squeeze().detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                # # s_show = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
                # s_show = np.stack([s, s, s], axis=-1)


                # transient = (transient.squeeze().detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                # # transient_show = cv2.cvtColor(transient, cv2.COLOR_GRAY2BGR)
                # transient_show = np.stack([transient, transient, transient], axis=-1)
                

                albedo_show = (albedo.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                irradiance_show = (irradiance.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)

                uncertainty = (uncertainty.squeeze().detach().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                uncertainty_show = cv2.cvtColor(uncertainty, cv2.COLOR_GRAY2BGR)

                row1 = np.concatenate([img_show, gt_img_show, depth_show], axis=1)
                row2 = np.concatenate([shadow_show, transient_show, s_show], axis=1)
                row3 = np.concatenate([irradiance_show, albedo_show, uncertainty_show], axis=1)
                image_to_show = np.concatenate([row1, row2, row3], axis=0)
                
                cv2.imwrite(os.path.join("./test_output", "%05d"%step + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

        return loss_dict
    
    @torch.no_grad()
    def eval_render(self, viewpoint_cam):
        self._gaussians.eval()
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)

        simp_outputs = self.simp_render_ortho(viewpoint_cam.sun_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp[:,:3])
        weights = simp_outputs['weights'].reshape(-1,1)
        counts = simp_outputs['counts'].reshape(-1,1)
        weights[counts <= 0] = 0.0
        weights[counts > 0] = weights[counts > 0] / counts[counts > 0]
        # weights = torch.nn.functional.softmax(weights)
        # weights = torch.sigmoid(weights)
        weights = torch.clip(weights, 0, 1)
        colors_precomp = torch.concatenate([colors_precomp, weights], dim=-1) #(N, 8)

        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)

        # update
        albedo = model_outputs["render"].to(self.device)
        transient = model_outputs["transient"].to(self.device)
        uncertainty = model_outputs["uncertainty"].to(self.device)
        shadow = model_outputs["shadow"].to(self.device)
        sky = model_outputs['sky_color'].to(self.device)

        s = shadow * transient
        irradiance = s + (1-s) * sky.reshape(3,1,1).repeat(1,s.shape[1],s.shape[2])
        image = albedo * irradiance

        model_outputs['render'] = image
        model_outputs['albedo'] = albedo
        model_outputs['shadow'] =shadow
        model_outputs['irradiance'] = irradiance
        model_outputs['uncertainty'] = uncertainty
        model_outputs['transient'] = transient
        return model_outputs
    
