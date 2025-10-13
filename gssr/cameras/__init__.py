#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
from dataclasses import dataclass, field
import torch
from torch import nn
import numpy as np
import torchvision.transforms as transforms

from gssr.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal

@dataclass
class CameraInfo():
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, resolution_scale=1.0,
                 zfar =100.0, znear=0.01,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device="cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.resolution_scale = resolution_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        ## only for PGSR ncc-loss
        self.gray_image = transforms.Grayscale()(self.original_image)
        self.near_ids = []
        self.ncc_scale = 1.0
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)
            self.gt_alpha_mask = None

        self.zfar = zfar
        self.znear = znear

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    # copied from PGSR
    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous()  # world2cam
        return intrinsic_matrix, extrinsic_matrix
    
    # copied from PGSR
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d
    
    # copied from PGSR
    def get_k(self, scale=1.0):
        K = torch.tensor([[self.Fx / scale, 0, self.Cx / scale],
                        [0, self.Fy / scale, self.Cy / scale],
                        [0, 0, 1]]).cuda()
        return K
    
    # copied from PGSR
    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale/self.Fx, 0, -self.Cx/self.Fx],
                            [0, scale/self.Fy, -self.Cy/self.Fy],
                            [0, 0, 1]]).cuda()
        return K_T


class MiniCam:
    def __init__(self, camera: Camera):
        self.uid = camera.uid
        self.image_width = camera.image_width
        self.image_height = camera.image_height
        self.FoVy = camera.FoVy
        self.FoVx = camera.FoVx
        self.znear = camera.znear
        self.zfar = camera.zfar
        self.resolution_scale = camera.resolution_scale
        self.world_view_transform = camera.world_view_transform
        self.full_proj_transform = camera.full_proj_transform
        self.projection_matrix = camera.projection_matrix
        self.camera_center = camera.camera_center
        self.gt_alpha_mask = camera.gt_alpha_mask


class OrthoCamera(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, ground_width, ground_height, 
                 resolution_scale=1.0, zfar=100, znear=0.01, trans=np.array([0, 0, 0]), scale=1, data_device="cuda"):
        super().__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, resolution_scale, zfar, znear, trans, scale, data_device)
        self.ground_width = ground_width
        self.ground_height = ground_height
        
        self.ortho_view = torch.tensor([1, 1, -1], dtype=torch.float32).cuda()

        # update Projection-Matrix & Full-proj-transform
        self.projection_matrix = torch.tensor([
            [2/(self.ground_width), 0, 0, 0],
            [0, 2/(self.ground_height), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float32).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


class SunCamera2(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, dx, dy, image, gt_alpha_mask, image_name, uid, resolution_scale=1, zfar=100, znear=0.01, trans=np.array([0, 0, 0]), scale=1, data_device="cuda"):
        super().__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, resolution_scale, zfar, znear, trans, scale, data_device)
        self.ground_width = dx
        self.ground_height = dy

        self.projection_matrix = torch.tensor([
            [2/dx, 0, 0, 0],
            [0, 2/dy, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float32).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)


class SunCamera(Camera):
    def __init__(self, uid, gsd, scene_bbx, sun_view, sun_dist=1e6):
        self.uid = uid

        ## Rotation and Translation
        sun_view = sun_view / np.linalg.norm(sun_view)
        sun_position = sun_view * sun_dist

        self.sun_view = torch.tensor(sun_view, dtype=torch.float32).cuda()
        self.sun_position = torch.tensor(sun_position, dtype=torch.float32).cuda()

        z_axis = -sun_view
        if np.isclose(np.abs(z_axis[1]), 1.0):
            up = np.array([1, 0, 0])
        else:
            up = np.array([0, 1, 0])

        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        R = np.stack((x_axis, y_axis, z_axis), axis=1)
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, -1] = sun_position
        w2c = np.linalg.inv(c2w)
        self.R = w2c[:3, :3].T
        self.T = w2c[:3, -1]

        ## bbx, compute dx dy
        mx = scene_bbx[0]
        my = scene_bbx[1]
        w = scene_bbx[2]
        h = scene_bbx[3]
        corner_points = np.array([
            [mx, my, 0.0, 1.0],
            [mx, my+h, 0.0, 1.0],
            [mx+w, my, 0.0, 1.0],
            [mx+w, my+h, 0.0, 1.0]
        ])
        corner_points_ic = (w2c @ corner_points.T).T[:, :3]
        dx = np.max(corner_points_ic[:,0]) - np.min(corner_points_ic[:,0])
        dy = np.max(corner_points_ic[:,1]) - np.min(corner_points_ic[:,1])

        self.image_width = int(dx / gsd + 0.5)
        self.image_height = int(dy / gsd + 0.5)

        if self.image_width > 1600:
            self.image_width = 1600
            dx = self.image_width * gsd
        if self.image_height > 1600:
            self.image_height = 1600
            dy = self.image_height * gsd

        h = np.sqrt((sun_position ** 2).sum())
        self.Fx = h / gsd
        self.Fy = h / gsd

        self.FoVx = focal2fov(self.Fx, self.image_width)
        self.FoVy = focal2fov(self.Fy, self.image_height)

        self.ground_width = dx
        self.ground_height = dy

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T)).transpose(0, 1).cuda()
        self.projection_matrix = torch.tensor([
            [2/dx, 0, 0, 0],
            [0, 2/dy, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=torch.float32).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class SatelliteCamera(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, 
                 mean_alt, scene_bbx, sun_view, sun_dist, resolution_scale=1, zfar=100, znear=0.01, trans=np.array([0, 0, 0]), scale=1, data_device="cuda"):
        super().__init__(colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid, resolution_scale, zfar, znear, trans, scale, data_device)
        
        self.mean_alt = mean_alt
        self.fx_z0 = self.Fx / self.mean_alt
        self.fy_z0 = self.Fy / self.mean_alt

        # setup suncamera
        if sun_view is not None:
            self.sun_camera = SunCamera(
                uid=self.uid,
                gsd=(1.0/self.fx_z0 + 1.0/self.fy_z0)/2.0,
                scene_bbx=scene_bbx,
                sun_view=sun_view,
                sun_dist=sun_dist
            )

        # self.projection_matrix = torch.tensor([
        #     [self.fx_z0 * (2 / self.image_width), 0, 0, 0],
        #     [0, self.fy_z0 * (2 / self.image_height), 0, 0],
        #     # [0, 0, 1, -self.mean_alt + 100.0],
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]], dtype=torch.float32).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
