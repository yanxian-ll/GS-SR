from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from torch._tensor import Tensor
import torch.nn.functional as F
import random
import math
import numpy as np
from pytorch3d.transforms import quaternion_to_matrix

from gssr.utils.point_utils import get_points_depth_in_depth_map, get_points_from_depth
from gssr.scene.vanilla_scene import VanillaScene, VanillaSceneConfig
from gssr.utils.graphics_utils import patch_offsets, patch_warp, normal_from_depth_image

from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer

@dataclass
class PGSRSceneConfig(VanillaSceneConfig):
    _target: type = field(default_factory=lambda: PGSRScene)
    lambda_normal: float = 0.015
    lambda_ncc: float = 0.15
    lambda_geo: float = 0.03
    patch_size: int = 3
    nunm_sample: int = 102400
    pixel_noise_threshold: float = 1.0

class PGSRScene(VanillaScene):
    config: PGSRSceneConfig

    # copied from PGSR
    def _get_img_grad_weight(self, img):
        hd, wd = img.shape[-2:]
        bottom_point = img[..., 2:hd,   1:wd-1]
        top_point    = img[..., 0:hd-2, 1:wd-1]
        right_point  = img[..., 1:hd-1, 2:wd]
        left_point   = img[..., 1:hd-1, 0:wd-2]
        grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
        grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
        grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
        grad_img, _ = torch.max(grad_img, dim=0)
        grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
        grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
        return grad_img
    
    # copied from PGSR
    def dilate(self, bin_img, ksize=5):
        pad = (ksize - 1) // 2
        bin_img = F.pad(bin_img, pad=[pad, pad, pad, pad], mode='reflect')
        out = F.max_pool2d(bin_img, kernel_size=ksize, stride=1, padding=0)
        return out

    # copied from PGSR
    def erode(self, bin_img, ksize=5):
        out = 1 - self.dilate(1 - bin_img, ksize)
        return out
    

    # copied from PGSR
    def lncc(self, ref, nea):
        # ref_gray: [batch_size, total_patch_size]
        # nea_grays: [batch_size, total_patch_size]
        bs, tps = nea.shape
        patch_size = int(np.sqrt(tps))

        ref_nea = ref * nea
        ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
        ref = ref.view(bs, 1, patch_size, patch_size)
        nea = nea.view(bs, 1, patch_size, patch_size)
        ref2 = ref.pow(2)
        nea2 = nea.pow(2)

        # sum over kernel
        filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
        padding = patch_size // 2
        ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
        nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
        ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
        nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
        ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

        # average over kernel
        ref_avg = ref_sum / tps
        nea_avg = nea_sum / tps

        cross = ref_nea_sum - nea_avg * ref_sum
        ref_var = ref2_sum - ref_avg * ref_sum
        nea_var = nea2_sum - nea_avg * nea_sum

        cc = cross * cross / (ref_var * nea_var + 1e-8)
        ncc = 1 - cc
        ncc = torch.clamp(ncc, 0.0, 2.0)
        ncc = torch.mean(ncc, dim=1, keepdim=True)
        mask = (ncc < 0.9)
        return ncc, mask
    
    def get_loss_dict(self, outputs, viewpoint_cam, step, metrics_dict=None, **kwargs) -> Dict[str, Tensor]:
        loss_dict = super().get_loss_dict(outputs, viewpoint_cam, step, metrics_dict, **kwargs)
        normal_loss = torch.tensor(0.0, device=self.device)
        ncc_loss = torch.tensor(0.0, device=self.device)
        geo_loss = torch.tensor(0.0, device=self.device)

        gt_image = viewpoint_cam.original_image.to(self.device)
        gt_image_gray = viewpoint_cam.gray_image.to(self.device)
        normal, depth_normal = outputs["rendered_normal"], outputs["depth_normal"]
        render_pkg = outputs
        # sigle-view loss
        if step > 7000:
            image_weight = (1.0 - self._get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 5
            image_weight = self.erode(image_weight[None, None]).squeeze()
            normal_loss = self.config.lambda_normal * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
        
        # multi-view loss
        if step > 7000:
            patch_size = self.config.patch_size
            total_patch_size = (self.config.patch_size * 2 + 1) ** 2
            ## compute geometry consistency mask and loss
            H, W = outputs['plane_depth'].squeeze().shape
            ix, iy = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
            pixels = torch.stack([ix, iy], dim=-1).float().to('cuda')

            near_cam = kwargs["near_cam"]
            nearest_render_pkg = kwargs["nearest_render_pkg"]

            pts = get_points_from_depth(viewpoint_cam, outputs['plane_depth'])
            pts_in_near_cam = pts @ near_cam.world_view_transform[:3,:3] + near_cam.world_view_transform[3,:3]
            map_z, d_mask = get_points_depth_in_depth_map(near_cam, nearest_render_pkg['plane_depth'], pts_in_near_cam)
                    
            pts_in_near_cam = pts_in_near_cam / (pts_in_near_cam[:,2:3])
            pts_in_near_cam = pts_in_near_cam * map_z.squeeze()[...,None]
            R = torch.tensor(near_cam.R).float().cuda()
            T = torch.tensor(near_cam.T).float().cuda()
            pts_ = (pts_in_near_cam - T) @ R.transpose(-1,-2)
            pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
            pts_projections = torch.stack(
                    [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                    pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
            pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
            d_mask = d_mask & (pixel_noise < self.config.pixel_noise_threshold)
            weights = (1.0 / torch.exp(pixel_noise)).detach()
            weights[~d_mask] = 0

            if d_mask.sum() > 0:
                geo_loss = self.config.lambda_geo * ((weights * pixel_noise)[d_mask]).mean()
                with torch.no_grad():
                    ## sample mask
                    d_mask = d_mask.reshape(-1)
                    valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                    if d_mask.sum() > self.config.nunm_sample:
                        index = np.random.choice(d_mask.sum().cpu().numpy(), self.config.nunm_sample, replace = False)
                        valid_indices = valid_indices[index]

                    weights = weights.reshape(-1)[valid_indices]  #(N,)
                    ## sample ref frame patch
                    pixels = pixels.reshape(-1,2)[valid_indices] #(N,2)
                    offsets = patch_offsets(patch_size, pixels.device) #(1,49,2)
                    ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float() #(N,49,2)
                                
                    H, W = gt_image_gray.squeeze().shape
                    pixels_patch = ori_pixels_patch.clone()  #(N,49,2)
                    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                    ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True) #(1,1,49N,1)
                    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size) #(N,49)

                    ref_to_neareast_r = near_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                    ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + near_cam.world_view_transform[3,:3]

                ## compute Homography
                ref_local_n = render_pkg["rendered_normal"].permute(1,2,0) #(H,W,3)
                ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]  #(N,3)

                ref_local_d = render_pkg['rendered_distance'].squeeze() #(H,W)
                ref_local_d = ref_local_d.reshape(-1)[valid_indices] #(N)
                
                H_ref_to_neareast = ref_to_neareast_r[None] - \
                    torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]  #(N,3,3)
                H_ref_to_neareast = torch.matmul(near_cam.get_k(near_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast) #(N,3,3)
                H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale) #(N,3,3)
                            
                ## compute neareast frame patch
                grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch) #(N,49,2)
                grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                nearest_image_gray = near_cam.gray_image.cuda()  #(1,H,W)
                sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True) #(1,1,49N,1)
                sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size) #(N, 49)
                            
                ## compute loss
                ncc, ncc_mask = self.lncc(ref_gray_val, sampled_gray_val) #(N,1)
                mask = ncc_mask.reshape(-1)
                ncc = ncc.reshape(-1) * weights
                ncc = ncc[mask].squeeze()

                if mask.sum() > 0:
                    ncc_loss = self.config.lambda_ncc * ncc.mean()
        
        loss_dict["normal_loss"] = normal_loss
        loss_dict["ncc_loss"] = ncc_loss
        loss_dict["geo_loss"] = geo_loss
        return loss_dict

    
    def get_train_loss_dict(self, step: int):
        viewpoint_cam = self.dataloader.next_train()

        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)
        metrics_dict = self.get_metrics_dict(model_outputs, viewpoint_cam)

        ## render near camera
        near_model_outputs = None
        near_cam = None
        if step > 7000:
            near_cam = None if len(viewpoint_cam.near_ids) == 0 else self.dataloader.getTrainData()[random.sample(viewpoint_cam.near_ids, 1)[0]]
            means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(near_cam)
            near_model_outputs = self.render(near_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
            near_model_outputs.update(other_output)

        loss_dict = self.get_loss_dict(model_outputs, viewpoint_cam, step, metrics_dict, near_cam=near_cam, nearest_render_pkg=near_model_outputs)
        return model_outputs, loss_dict, metrics_dict
    

    def render_normal(self, viewpoint_cam, depth, offset=None, normal=None, scale=1):
        # depth: (H, W), bg_color: (3), alpha: (H, W)
        # normal_ref: (3, H, W)
        intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
        st = max(int(scale/2)-1,0)
        if offset is not None: offset = offset[st::scale,st::scale]
        normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                                intrinsic_matrix.to(depth.device), 
                                                extrinsic_matrix.to(depth.device), offset)

        normal_ref = normal_ref.permute(2,0,1)
        return normal_ref
    

    def get_rotation_matrix(self, rotation):
        return quaternion_to_matrix(rotation)

    def get_smallest_axis(self, rotation, scaling, return_idx=False):
        rotation_matrices = self.get_rotation_matrix(rotation)
        smallest_axis_idx = scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    
    def get_normal(self, view_cam, xyz, rotation, scaling):
        normal_global = self.get_smallest_axis(rotation, scaling)
        gaussian_to_cam_global = view_cam.camera_center - xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    

    def render(self, viewpoint_camera, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp):
         # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        screenspace_points_abs = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
            screenspace_points_abs.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        means2D = screenspace_points
        means2D_abs = screenspace_points_abs

        raster_settings = PlaneGaussianRasterizationSettings(
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
                render_geo=True,
                debug=self.config.debug
            )

        rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

        global_normal = self.get_normal(viewpoint_camera, means3D, rotations, scales)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        rendered_image, radii, out_observe, out_all_map, plane_depth = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            all_map = input_all_map,
            cov3D_precomp = cov3D_precomp)

        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4, ]
        rendered_distance = out_all_map[4:5, ]

        depth_normal = self.render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe,
                        "rendered_normal": rendered_normal,
                        "plane_depth": plane_depth,
                        "rendered_distance": rendered_distance,
                        "depth_normal": depth_normal,

                        "normal": rendered_normal,
                        "depth": plane_depth,
                        }
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return return_dict