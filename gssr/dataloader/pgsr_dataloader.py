import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Type

from gssr.dataloader.colmap_dataloader import ColmapDataLoader, ColmapDataLoaderConfig
from gssr.dataloader.depth_prior import load_murre_depth_point_cloud
from gssr.utils.mvsnet_utils import read_pairs, write_pairs, read_model, qvec2rotmat, view_selection

from rich.console import Console
CONSOLE = Console(width=120)


@dataclass
class PGSRDataLoaderConfig(ColmapDataLoaderConfig):
    _target: Type = field(default_factory=lambda: PGSRDataLoader)
    use_mvs_view_selection: bool = False
    num_multi_view: int = 5
    multi_view_max_angle: float = 30.0
    multi_view_min_dis: float = 0.01
    multi_view_max_dis: float = 1.5

    # Optional dense depth prior for initialization. Disabled by default so the
    # original SfM point-cloud initialization remains unchanged.
    depth_init_dir: Optional[str] = None
    """Murre depth directory, relative to source_dir or absolute. Example: murre_depth."""
    depth_init_num_points: int = 100_000
    """Uniformly sample this many valid depth-derived 3D points; <=0 keeps all."""
    depth_init_seed: int = 0
    depth_init_min_depth: float = 1e-6
    depth_init_max_depth: Optional[float] = None
    """Optional camera-Z cutoff in original COLMAP units; None uses Murre manifest max_depth."""


class PGSRDataLoader(ColmapDataLoader):
    config: PGSRDataLoaderConfig

    def __init__(self, config: PGSRDataLoaderConfig, source_dir: str, eval: bool = False, world_size: int = 1, local_rank: int = 0):
        super().__init__(config, source_dir, eval, world_size, local_rank)

        if self.config.depth_init_dir:
            train_image_names = [cam.image_name for cam in self.getTrainData()]
            scene_scale = 1.0 if self.config.scene_scale is None else float(self.config.scene_scale)
            scene_translation = tuple(
                0.0 if value is None else float(value)
                for value in (self.config.t_x, self.config.t_y, self.config.t_z)
            )
            self.point_cloud, depth_stats = load_murre_depth_point_cloud(
                source_dir=self.source_dir,
                depth_dir=self.config.depth_init_dir,
                train_image_names=train_image_names,
                num_points=self.config.depth_init_num_points,
                seed=self.config.depth_init_seed,
                scale_scene=self.config.scale_scene,
                scene_scale=scene_scale,
                scene_translation=scene_translation,
                depth_min=self.config.depth_init_min_depth,
                depth_max=self.config.depth_init_max_depth,
            )
            CONSOLE.log(
                "Depth-prior initialization replaces SfM points: "
                f"{depth_stats.num_frames} train views, "
                f"{depth_stats.num_valid_points} valid depth points -> "
                f"{depth_stats.num_sampled_points} sampled points "
                f"from {depth_stats.depth_dir}"
            )

        ## View Selection
        self.num_multi_view = self.config.num_multi_view
        if os.path.exists(os.path.join(self.source_dir, 'pair.txt')):
            view_sel = read_pairs(os.path.join(self.source_dir, 'pair.txt'))

        elif self.config.use_mvs_view_selection and os.path.exists(os.path.join(self.source_dir, "sparse")):   # (only support colmap format)
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
