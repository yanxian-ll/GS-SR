import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Type

from gssr.dataloader.colmap_dataloader import ColmapDataLoader, ColmapDataLoaderConfig
from gssr.utils.mvsnet_utils import read_pairs, write_pairs, read_model, qvec2rotmat, view_selection

from rich.console import Console
CONSOLE = Console(width=120)

@dataclass
class PGSRDataLoaderConfig(ColmapDataLoaderConfig):
    _target: Type = field(default_factory=lambda: PGSRDataLoader)
    num_multi_view: int = 5
    multi_view_max_angle: float = 30.0
    multi_view_min_dis: float = 0.01
    multi_view_max_dis: float = 1.5


class PGSRDataLoader(ColmapDataLoader):
    config: PGSRDataLoaderConfig

    def __init__(self, config: ColmapDataLoaderConfig, source_dir: str, eval: bool = False, world_size: int = 1, local_rank: int = 0):
        super().__init__(config, source_dir, eval, world_size, local_rank)

        ## View Selection 
        self.num_multi_view = self.config.num_multi_view
        if os.path.exists(os.path.join(self.source_dir, 'pair.txt')):
            view_sel = read_pairs(os.path.join(self.source_dir, 'pair.txt'))

        elif os.path.exists(os.path.join(self.source_dir, "sparse")):   # (only support colmap format)
            CONSOLE.log("Start View Selection.")
            _, extr_infos, points3d = read_model(os.path.join(self.source_dir, 'sparse/0'))
            list_extr_infos = [extr_infos[cam.colmap_id] for cam in self.train_dataset[1.0]]
            list_cam_centers = []
            list_point3d_ids = []
            for extr in list_extr_infos:
                R = qvec2rotmat(extr.qvec)
                t = np.array(extr.tvec).reshape(3,1)
                center = -np.matmul(R.T, t)[:, 0]
                point3d_ids = extr.point3D_ids
                list_cam_centers.append(center)
                list_point3d_ids.append(point3d_ids)
            view_sel = view_selection(list_cam_centers, list_point3d_ids, points3d, num_views=self.num_multi_view)
            write_pairs(os.path.join(self.source_dir, 'pair.txt'), view_sel)

        else:  # copied from PGSR
            camera_centers, center_rays, world_view_transforms = [], [], []
            for id, cur_cam in enumerate(self.train_dataset[1.0]):
                world_view_transforms.append(cur_cam.world_view_transform)
                camera_centers.append(cur_cam.camera_center)
                R = torch.tensor(cur_cam.R).float().cuda()
                T = torch.tensor(cur_cam.T).float().cuda()
                center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
                center_ray = center_ray @ R.transpose(-1,-2)
                center_rays.append(center_ray)

            world_view_transforms = torch.stack(world_view_transforms)
            camera_centers = torch.stack(camera_centers, dim=0)
            center_rays = torch.stack(center_rays, dim=0)
            center_rays = torch.nn.functional.normalize(center_rays, dim=-1)
            diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
            tmp = torch.sum(center_rays[:,None] * center_rays[None], dim=-1)
            angles = torch.arccos(tmp) * 180 / 3.1415926
            angles = angles.detach().cpu().numpy()

            view_sel = []
            for id, cur_cam in enumerate(self.train_dataset[1.0]):
                sorted_indices = np.lexsort((angles[id], diss[id]))
                mask = (angles[id][sorted_indices] < self.config.multi_view_max_angle) & \
                       (diss[id][sorted_indices] > self.config.multi_view_min_dis) & \
                       (diss[id][sorted_indices] < self.config.multi_view_max_dis)
                sorted_indices = sorted_indices[mask]
                multi_view_num = min(self.num_multi_view, len(sorted_indices))
                view_sel.append([(k, angles[id][k]) for k in sorted_indices[:multi_view_num]])

            write_pairs(os.path.join(self.source_dir, 'pair.txt'), view_sel)


        for resolution_scale in self.config.resolution_scales:
            for i, cam in enumerate(self.train_dataset[resolution_scale]):
                cam.near_ids = [k for k, s in view_sel[i]]

