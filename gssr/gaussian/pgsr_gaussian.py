import torch.nn as nn
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast

from gssr.gaussian.vanilla_gaussian import VanillaGaussian, VanillaGaussianConfig, build_rotation
from gssr.utils.graphics_utils import BasicPointCloud


@dataclass
class PGSRGaussianConfig(VanillaGaussianConfig):
    _target: type = field(default_factory=lambda: PGSRGaussian)
    densify_abs_grad_threshold: float = 0.0008
    abs_split_radii2D_threshold: float = 20
    max_abs_split_points: int = 50_000
    max_all_points: int = 6000_000


class PGSRGaussian(VanillaGaussian):
    config: PGSRGaussianConfig

    def __init__(self, config: VanillaGaussianConfig, device: str = 'cuda', world_size: int = 1, local_rank: int = 0) -> None:
        super().__init__(config, device, world_size, local_rank)
        self.abs_split_radii2D_threshold = self.config.abs_split_radii2D_threshold
        self.max_abs_split_points = self.config.max_abs_split_points
        self.max_all_points = self.config.max_all_points

        self.max_weight = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom_abs = torch.empty(0)
    
    def create_from_data(self, pcd: BasicPointCloud, cameras: Dict, spatial_lr_scale: float):
        super().create_from_data(pcd, cameras, spatial_lr_scale)
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device=self.device)
    
    def setup_optimizers(self):
        super().setup_optimizers()
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
    
    def prune_points(self, mask):
        valid_points_mask = super().prune_points(mask)
        self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]
        self.denom_abs = self.denom_abs[valid_points_mask]
        self.max_weight = self.max_weight[valid_points_mask]
        return valid_points_mask
    
    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        super().densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.denom_abs = torch.zeros((self.get_xyz.shape[0], 1), device=self.device)
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device=self.device)
    
    def densify_and_split(self, grads, grad_threshold, grads_abs, grad_abs_threshold, scene_extent, max_radii2D, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_grads_abs = torch.zeros((n_init_points), device=self.device)
        padded_grads_abs[:grads_abs.shape[0]] = grads_abs.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points), device=self.device)
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D.squeeze()

        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            padded_grad[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(padded_grad, (1.0-ratio))
            selected_pts_mask = torch.where(padded_grad > threshold, True, False)
            # print(f"split {selected_pts_mask.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")
        else:
            padded_grads_abs[selected_pts_mask] = 0
            mask = (torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent) & (padded_max_radii2D > self.abs_split_radii2D_threshold)
            padded_grads_abs[~mask] = 0
            selected_pts_mask_abs = torch.where(padded_grads_abs >= grad_abs_threshold, True, False)
            limited_num = min(self.max_all_points - n_init_points - selected_pts_mask.sum(), self.max_abs_split_points)
            if selected_pts_mask_abs.sum() > limited_num:
                ratio = limited_num / float(n_init_points)
                threshold = torch.quantile(padded_grads_abs, (1.0-ratio))
                selected_pts_mask_abs = torch.where(padded_grads_abs > threshold, True, False)
            selected_pts_mask = torch.logical_or(selected_pts_mask, selected_pts_mask_abs)
            # print(f"split {selected_pts_mask.sum()}, abs {selected_pts_mask_abs.sum()}, raddi2D {padded_max_radii2D.max()} ,{padded_max_radii2D.median()}")

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device=self.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
    
    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if selected_pts_mask.sum() + n_init_points > self.max_all_points:
            limited_num = self.max_all_points - n_init_points
            grads_tmp = grads.squeeze().clone()
            grads_tmp[~selected_pts_mask] = 0
            ratio = limited_num / float(n_init_points)
            threshold = torch.quantile(grads_tmp, (1.0-ratio))
            selected_pts_mask = torch.where(grads_tmp > threshold, True, False)

        if selected_pts_mask.sum() > 0:
            # print(f"clone {selected_pts_mask.sum()}")
            new_xyz = self._xyz[selected_pts_mask]

            stds = self.get_scaling[selected_pts_mask]
            means =torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(self._rotation[selected_pts_mask])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask]
            
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]

            self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, abs_max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads_abs = self.xyz_gradient_accum_abs / self.denom_abs
        grads[grads.isnan()] = 0.0
        grads_abs[grads_abs.isnan()] = 0.0
        max_radii2D = self.max_radii2D.clone()

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, grads_abs, abs_max_grad, extent, max_radii2D)

        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        # print(f"all points {self._xyz.shape[0]}")
        torch.cuda.empty_cache()
    
    def add_densification_stats(self, viewspace_point_tensor, viewspace_point_tensor_abs, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor_abs.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        self.denom_abs[update_filter] += 1

    
    def densify(self, step, **kwargs):
        visibility_filter, radii, viewspace_points = kwargs["visibility_filter"], kwargs["radii"], kwargs["viewspace_points"]
        out_observe = kwargs["out_observe"]
        viewspace_points_abs = kwargs["viewspace_points_abs"]
        if step < self.config.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            mask = (out_observe > 0) & visibility_filter
            self.max_radii2D[mask] = torch.max(self.max_radii2D[mask], radii[mask])
            self.add_densification_stats(viewspace_points, viewspace_points_abs, visibility_filter)

            if step > self.config.densify_from_iter and step % self.config.densification_interval == 0:
                size_threshold = 20 if step > self.config.opacity_reset_interval else None
                self.densify_and_prune(self.config.densify_grad_threshold, 
                                       self.config.densify_abs_grad_threshold, 
                                       self.config.opacity_cull_threshold, 
                                       self.spatial_lr_scale, size_threshold)
            # reset_opacity                    
            if step % self.config.opacity_reset_interval == 0:
                self.reset_opacity()