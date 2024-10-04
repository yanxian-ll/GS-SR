from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from torch._tensor import Tensor
import math
from einops import repeat

from gssr.scene.base_scene import SceneConfig
from gssr.utils.point_utils import depth_to_normal
from gssr.scene.scaffold_scene import ScaffoldScene, ScaffoldSceneConfig, GaussianRasterizationSettings, GaussianRasterizer

@dataclass
class OctreeSceneConfig(ScaffoldSceneConfig):
    _target: type = field(default_factory=lambda: OctreeScene)
    coarse_iter: int = 10000
    coarse_factor: float = 1.5


class OctreeScene(ScaffoldScene):
    config: OctreeSceneConfig

    def __init__(self, config: SceneConfig, source_dir: str, eval: bool = False, device: str = 'cuda', world_size: int = 1, local_rank: int = 0) -> None:
        super().__init__(config, source_dir, eval, device, world_size, local_rank)
        self._gaussians.set_coarse_interval(self.config.coarse_iter, self.config.coarse_factor)
    
    def generate_neural_gaussians(self, viewpoint_camera, visible_mask=None, is_training=False):
        ## view frustum filtering for acceleration    
        if visible_mask is None:
            visible_mask = torch.ones(self._gaussians.get_anchor.shape[0], dtype=torch.bool, device = self._gaussians.get_anchor.device)

        anchor = self._gaussians.get_anchor[visible_mask]
        feat = self._gaussians._anchor_feat[visible_mask]
        level = self._gaussians.get_level[visible_mask]
        grid_offsets = self._gaussians._offset[visible_mask]
        grid_scaling = self._gaussians.get_scaling[visible_mask]

        ## get view properties for anchor
        ob_view = anchor - viewpoint_camera.camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        ## view-adaptive feature
        if self._gaussians.use_feat_bank:
            if self._gaussians.add_level:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view
            
            bank_weight = self._gaussians.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

            ## multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                feat[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1) # [n, c]

        if self._gaussians.add_level:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1) # [N, c+3+1+1]
            cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1) # [N, c+3+1]
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
        
        if self._gaussians.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
            appearance = self._gaussians.get_appearance(camera_indicies)
                
        # get offset's opacity
        if self._gaussians.add_opacity_dist:
            neural_opacity = self._gaussians.get_opacity_mlp(cat_local_view) # [N, k]
        else:
            neural_opacity = self._gaussians.get_opacity_mlp(cat_local_view_wodist)
        
        if self._gaussians.dist2level=="progressive":
            prog = self._gaussians._prog_ratio[visible_mask]
            transition_mask = self._gaussians.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity>0.0)
        mask = mask.view(-1)

        # select opacity 
        opacity = neural_opacity[mask]

        # get offset's color
        if self._gaussians.appearance_dim > 0:
            if self._gaussians.add_color_dist:
                color = self._gaussians.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
            else:
                color = self._gaussians.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
        else:
            if self._gaussians.add_color_dist:
                color = self._gaussians.get_color_mlp(cat_local_view)
            else:
                color = self._gaussians.get_color_mlp(cat_local_view_wodist)
        color = color.reshape([anchor.shape[0]*self._gaussians.n_offsets, 3])# [mask]

        # get offset's cov
        if self._gaussians.add_cov_dist:
            scale_rot = self._gaussians.get_cov_mlp(cat_local_view)
        else:
            scale_rot = self._gaussians.get_cov_mlp(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([anchor.shape[0]*self._gaussians.n_offsets, 7]) # [mask]
    
        # offsets
        offsets = grid_offsets.view([-1, 3]) # [mask]
        
        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, anchor], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self._gaussians.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
        
        # post-process cov
        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
        rot = self._gaussians.rotation_activation(scale_rot[:,3:7])
        
        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets 

        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask
        else:
            return xyz, color, opacity, scaling, rot
    

    def prefilter_voxel(self, viewpoint_camera):
        """Render the scene. """
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

        anchor_mask = self._gaussians._anchor_mask
        means3D = self._gaussians.get_anchor[anchor_mask]
        cov3D_precomp = None
        scales = self._gaussians.get_scaling[anchor_mask]
        rotations = self._gaussians.get_rotation[anchor_mask]

        radii_pure = rasterizer.visible_filter(
            means3D = means3D,
            scales = scales[:,:3],
            rotations = rotations,
            cov3D_precomp = cov3D_precomp
        )
        visible_mask = anchor_mask.clone()
        visible_mask[anchor_mask] = radii_pure > 0
        return visible_mask

    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict."""
        viewpoint_cam = self.dataloader.next_train()
        
        self._gaussians.set_anchor_mask(viewpoint_cam.camera_center, step, viewpoint_cam.resolution_scale)
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)

        metrics_dict = self.get_metrics_dict(model_outputs, viewpoint_cam)
        loss_dict = self.get_loss_dict(model_outputs, viewpoint_cam, step, metrics_dict)
        return model_outputs, loss_dict, metrics_dict
    
    @torch.no_grad()
    def eval_render(self, viewpoint_cam):
        self._gaussians.eval()
        self._gaussians.set_anchor_mask(viewpoint_cam.camera_center, 0, viewpoint_cam.resolution_scale)
        means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, other_output = self.generate_gaussians(viewpoint_cam)
        model_outputs = self.render(viewpoint_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)
        model_outputs.update(other_output)
        return model_outputs
    