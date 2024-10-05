import torch.nn as nn
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, Union, cast
import numpy as np
from functools import reduce
from torch_scatter import scatter_max
import os
from einops import repeat

from gssr.engine.callbacks import TrainingCallback
from gssr.gaussian.vanilla_gaussian import (
    VanillaGaussian, 
    VanillaGaussianConfig, 
    TrainingCallbackLocation,
    distCUDA2, 
    get_expon_lr_func,
    PlyElement, PlyData)
from gssr.utils.graphics_utils import BasicPointCloud
from gssr.gaussian.embedding import Embedding


@dataclass
class ScaffoldGaussianConfig(VanillaGaussianConfig):
    _target: type = field(default_factory=lambda: ScaffoldGaussian)

    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001  # if voxel_size<=0, using 1nn dist
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4

    start_stat: int = 500  # for anchor densification
    densification_interval: int = 100
    densify_from_iter: int = 1500
    densify_until_iter: int = 15_000
    success_threshold: float = 0.8

    use_feat_bank: bool = False
    appearance_dim: int = 32

    view_dim: int = 3
    
    # In the Bungeenerf dataset, we propose to set the following three parameters to True,
    # Because there are enough dist variations.
    add_opacity_dist : bool = False
    add_cov_dist : bool = False
    add_color_dist : bool = False

    ## learning rate
    position_lr_init: float = 0.0
    position_lr_final: float = 0.0
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000

    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002
        
    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000

    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002  
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000

    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 30_000
        
    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000
    
    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 30_000

    appearance_lr_init: float = 0.05
    appearance_lr_final: float = 0.0005
    appearance_lr_delay_mult: float = 0.01
    appearance_lr_max_steps: int = 30_000

    save_ckpt_mode = 'split'
    """split or unit"""
    

class ScaffoldGaussian(VanillaGaussian):
    config: ScaffoldGaussianConfig

    def __init__(self, config: VanillaGaussianConfig, device: str = 'cuda', world_size: int = 1, local_rank: int = 0) -> None:
        super().__init__(config, device, world_size, local_rank)
        self.active_sh_degree = 1

        self.feat_dim = self.config.feat_dim
        self.n_offsets = self.config.n_offsets
        self.voxel_size = self.config.voxel_size
        self.update_depth = self.config.update_depth
        self.update_init_factor = self.config.update_init_factor
        self.update_hierachy_factor = self.config.update_hierachy_factor
        self.use_feat_bank = self.config.use_feat_bank

        self.appearance_dim = self.config.appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = self.config.add_opacity_dist
        self.add_cov_dist = self.config.add_cov_dist
        self.add_color_dist = self.config.add_color_dist

        self._anchor = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)

        self.opacity_accum = torch.empty(0)

        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.anchor_demon = torch.empty(0)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0

        self.view_dim = self.config.view_dim

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(self.view_dim + 1, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3),
                nn.Softmax(dim=1)
            ).to(device)

        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, self.n_offsets),
            nn.Tanh()
        ).to(device)

        self.mlp_cov = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 7*self.n_offsets),
        ).to(device)

        self.mlp_color = nn.Sequential(
            nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim, self.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.feat_dim, 3*self.n_offsets),
            nn.Sigmoid()
        ).to(device)
    
    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if self.use_feat_bank:                   
            self.mlp_feature_bank.train()


    def capture(self):
        return (
            self._anchor,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args):
        (self.active_sh_degree, 
        self._anchor, 
        self._offset,
        self._local,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        # self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
    
    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).to(self.device)
    
    @property
    def get_appearance(self):
        return self.embedding_appearance
    
    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)
    
    @property
    def get_featurebank_mlp(self):
        return self.mlp_feature_bank
    
    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity
    
    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_anchor(self):
        return self._anchor
    
    @property
    def set_anchor(self, new_anchor):
        assert self._anchor.shape == new_anchor.shape
        del self._anchor
        torch.cuda.empty_cache()
        self._anchor = new_anchor
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data/voxel_size), axis=0)*voxel_size    
        return data
    
    def create_from_data(self, pcd: BasicPointCloud, cameras: Dict, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = pcd.points[::self.config.sampling_ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0]*0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        print(f'Initial voxel_size: {self.voxel_size}')
        
        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((fused_point_cloud.shape[0], self.feat_dim)).float().cuda()
        
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._anchor = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")
        
    def setup_optimizers(self):
        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")
        
        l = [
            {'params': [self._anchor], 'lr': self.config.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': self.config.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': self.config.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': self.config.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': self.config.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': self.config.rotation_lr, "name": "rotation"},
                
            {'params': self.mlp_opacity.parameters(), 'lr': self.config.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': self.config.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': self.config.mlp_color_lr_init, "name": "mlp_color"},
        ]
        if self.appearance_dim > 0:
            l.append(
                {'params': self.embedding_appearance.parameters(), 'lr': self.config.appearance_lr_init, "name": "embedding_appearance"}
            )
        if self.use_feat_bank:
            l.append(
                {'params': self.mlp_feature_bank.parameters(), 'lr': self.config.mlp_featurebank_lr_init, "name": "mlp_featurebank"}
            )

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=self.config.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.config.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.config.position_lr_delay_mult,
                                                    max_steps=self.config.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=self.config.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=self.config.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=self.config.offset_lr_delay_mult,
                                                    max_steps=self.config.offset_lr_max_steps)
        
        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=self.config.mlp_opacity_lr_init,
                                                    lr_final=self.config.mlp_opacity_lr_final,
                                                    lr_delay_mult=self.config.mlp_opacity_lr_delay_mult,
                                                    max_steps=self.config.mlp_opacity_lr_max_steps)
        
        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=self.config.mlp_cov_lr_init,
                                                    lr_final=self.config.mlp_cov_lr_final,
                                                    lr_delay_mult=self.config.mlp_cov_lr_delay_mult,
                                                    max_steps=self.config.mlp_cov_lr_max_steps)
        
        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=self.config.mlp_color_lr_init,
                                                    lr_final=self.config.mlp_color_lr_final,
                                                    lr_delay_mult=self.config.mlp_color_lr_delay_mult,
                                                    max_steps=self.config.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=self.config.mlp_featurebank_lr_init,
                                                        lr_final=self.config.mlp_featurebank_lr_final,
                                                        lr_delay_mult=self.config.mlp_featurebank_lr_delay_mult,
                                                        max_steps=self.config.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=self.config.appearance_lr_init,
                                                        lr_final=self.config.appearance_lr_final,
                                                        lr_delay_mult=self.config.appearance_lr_delay_mult,
                                                        max_steps=self.config.appearance_lr_max_steps)

    def update_learning_rate(self, step):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(step)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(step)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(step)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(step)
                param_group['lr'] = lr
    

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_gaussians(self, path: str):
        anchor = self._anchor.detach().cpu().numpy()
        normals = np.zeros_like(anchor)
        anchor_feat = self._anchor_feat.detach().cpu().numpy()
        offset = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, normals, offset, anchor_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_gaussians(self, path):
        plydata = PlyData.read(path)

        anchor = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1).astype(np.float32)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        
        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata.elements[0][attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))
        
        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))

        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    # statis grad information to guide liftting. 
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0
        
        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)
        
        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter
        
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1
        
    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_anchor(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)
        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        return valid_points_mask

    
    def anchor_growing(self, grads, threshold, offset_mask):
        init_length = self.get_anchor.shape[0]*self.n_offsets
        for i in range(self.update_depth):
            # update threshold
            cur_threshold = threshold*((self.update_hierachy_factor//2)**i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)
            
            # random pick
            rand_mask = torch.rand_like(candidate_mask.float())>(0.5**(i+1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)
            
            length_inc = self.get_anchor.shape[0]*self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)
            
            # assert self.update_init_factor // (self.update_hierachy_factor**i) > 0
            # size_factor = min(self.update_init_factor // (self.update_hierachy_factor**i), 1)
            size_factor = self.update_init_factor // (self.update_hierachy_factor**i)
            cur_size = self.voxel_size*size_factor
            
            grid_coords = torch.round(self.get_anchor / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)


            ## split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)
                
                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size

            
            if candidate_anchor.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], device=candidate_anchor.device).float()
                new_rotation[:,0] = 1.0

                new_opacities = self.inverse_opacity_activation(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))

                new_feat = self._anchor_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.feat_dim])[candidate_mask]

                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()

                d = {
                    "anchor": candidate_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }
                

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()
                
                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                
    def adjust_anchor(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)
        
        self.anchor_growing(grads_norm, grad_threshold, offset_mask)
        
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32, 
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)
        
        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1) # [N, 1]
        prune_mask = torch.logical_and(prune_mask, anchors_mask) # [N] 
        
        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum
        
        # update opacity accum 
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
        
        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)
        
        self.max_radii2D = torch.zeros((self.get_anchor.shape[0]), device="cuda")

    def densify(self, step, **kwargs):
        viewspace_points, visibility_filter, offset_selection_mask, opacity, voxel_visible_mask = kwargs["viewspace_points"], kwargs["visibility_filter"], kwargs["selection_mask"], kwargs["neural_opacity"], kwargs["voxel_visible_mask"]
        # densification
        if step < self.config.densify_until_iter and step > self.config.start_stat:
            # add statis
            self.training_statis(viewspace_points, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
            # densification
            if step > self.config.densify_from_iter and step % self.config.densification_interval == 0:
                self.adjust_anchor(check_interval=self.config.densification_interval, 
                                   success_threshold=self.config.success_threshold, 
                                   grad_threshold=self.config.densify_grad_threshold, 
                                   min_opacity=self.config.opacity_cull_threshold)
        elif step == self.config.densify_until_iter:
            del self.opacity_accum
            del self.offset_gradient_accum
            del self.offset_denom
            torch.cuda.empty_cache()
    
    def get_training_callbacks(self) -> List[TrainingCallback]:
        callbacks = [
            TrainingCallback(
                label="update_learning_rate",
                where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=self.update_learning_rate,
            )
        ]
        return callbacks


    def save_mlp_checkpoints(self, path):
        if self.config.save_ckpt_mode == 'split':
            self.mlp_opacity.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+3+self.opacity_dist_dim).cuda()))
            opacity_mlp.save(os.path.join(path, 'opacity_mlp.pt'))
            self.mlp_opacity.train()

            self.mlp_cov.eval()
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+3+self.cov_dist_dim).cuda()))
            cov_mlp.save(os.path.join(path, 'cov_mlp.pt'))
            self.mlp_cov.train()

            self.mlp_color.eval()
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+3+self.color_dist_dim+self.appearance_dim).cuda()))
            color_mlp.save(os.path.join(path, 'color_mlp.pt'))
            self.mlp_color.train()

            if self.use_feat_bank:
                self.mlp_feature_bank.eval()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+1).cuda()))
                feature_bank_mlp.save(os.path.join(path, 'feature_bank_mlp.pt'))
                self.mlp_feature_bank.train()

            if self.appearance_dim:
                self.embedding_appearance.eval()
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, 'embedding_appearance.pt'))
                self.embedding_appearance.train()

        elif self.config.save_ckpt_mode == 'unite':
            if self.use_feat_bank:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'feature_bank_mlp': self.mlp_feature_bank.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            elif self.appearance_dim > 0:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    'appearance': self.embedding_appearance.state_dict()
                    }, os.path.join(path, 'checkpoints.pth'))
            else:
                torch.save({
                    'opacity_mlp': self.mlp_opacity.state_dict(),
                    'cov_mlp': self.mlp_cov.state_dict(),
                    'color_mlp': self.mlp_color.state_dict(),
                    }, os.path.join(path, 'checkpoints.pth'))
        else:
            raise NotImplementedError


    def load_mlp_checkpoints(self, path):#split or unite
        if self.config.save_ckpt_mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, 'opacity_mlp.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, 'cov_mlp.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, 'color_mlp.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, 'feature_bank_mlp.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, 'embedding_appearance.pt')).cuda()
        elif self.config.save_ckpt_mode == 'unite':
            checkpoint = torch.load(os.path.join(path, 'checkpoints.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    

    