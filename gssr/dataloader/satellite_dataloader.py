import os
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Type, Optional

from gssr.dataloader.colmap_dataloader import ColmapDataLoader, ColmapDataLoaderConfig
from gssr.pointcloud import BasicPointCloud
from gssr.cameras import SatelliteCamera
from gssr.utils.graphics_utils import focal2fov
from gssr.utils.mvsnet_utils import read_pairs, write_pairs

from rich.console import Console
CONSOLE = Console(width=120)


# read IMD file with sensor azimuth, elevation, view-angle
def read_imd(name):
    file = open(name, 'r')
    lines = file.readlines()
    for j in range(len(lines)):
        pos = lines[j].find('meanSatAz')
        if pos != -1:
            last = lines[j].find(';') - 1
            az = float(lines[j][pos + 11:last])

        pos = lines[j].find('meanSatEl')
        if pos != -1:
            last = lines[j].find(';') - 1
            el = float(lines[j][pos + 11:last])
        
        pos = lines[j].find('meanSunAz')
        if pos != -1:
            last = lines[j].find(';') - 1
            sun_az = float(lines[j][pos + 11:last])

        pos = lines[j].find('meanSunEl')
        if pos != -1:
            last = lines[j].find(';') - 1
            sun_el = float(lines[j][pos + 11:last])
    return az, el, sun_az, sun_el

def sun_angles_to_enu(azimuth, elevation):
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)
    E = np.cos(elevation_rad) * np.sin(azimuth_rad)
    N = np.cos(elevation_rad) * np.cos(azimuth_rad)
    U = np.sin(elevation_rad)
    dir = np.array([E, N, U])
    return dir / np.linalg.norm(dir)

@dataclass
class SatelliteDataLoaderConfig(ColmapDataLoaderConfig):
    _target: Type = field(default_factory=lambda: SatelliteDataLoader)
    llffhold: int = -1
    # following satenerf, split train / val images
    val_percent: float = 0.15
    min_val: int = 2
    # whether to add water points
    add_water_points: bool = False

@dataclass
class SatelliteDataLoader(ColmapDataLoader):
    '''DataLoader for satellite images, inherits from ColmapDataLoader
    '''
    config: SatelliteDataLoaderConfig

    def __init__(self, config: SatelliteDataLoaderConfig, source_dir:str, 
                 eval:bool = False, world_size:int = 1, local_rank:int = 0):
        # following satenerf, split train / val images
        if eval:
            image_path = os.path.join(source_dir, config.images)
            num_images = len(os.listdir(image_path))
            num_val = max(int(num_images * config.val_percent), config.min_val)
            config.llffhold = num_images // num_val
            CONSOLE.log(f"Num-images: {num_images}, Num-train: {num_images - num_val}, \
                        Num-val: {num_val}, compute-llffhold: {config.llffhold}")

        # run the original init function
        super().__init__(config, source_dir, eval, world_size, local_rank)

        ## TODO: find a better value (用于控制学习率大小)
        self.cameras_extent = 2.0

        ## add water points
        if config.add_water_points:
            pc = self.point_cloud
            mz, Mz = np.percentile(pc.points[:, -1], 5), np.percentile(pc.points[:, -1], 95)
            mask = (pc.points[:, -1]>mz) & (pc.points[:, -1]<Mz)
            xyz = pc.points[mask]

            counts, bin_edges = np.histogram(xyz[:, -1], bins=100)
            max_count_index = np.argmax(counts)
            bottom_z = (bin_edges[max_count_index] + bin_edges[max_count_index+1]) / 2

            xyz *= self.config.scene_scale  # to original scale(meter)
            max_x, min_x = np.max(xyz[:,0]), np.min(xyz[:,0])
            max_y, min_y = np.max(xyz[:,1]), np.min(xyz[:,1])

            grid_size = 2  # meter
            gridx, gridy = np.meshgrid(np.linspace(min_x, max_x, int((max_x-min_x)//grid_size+1)),
                                    np.linspace(min_y, max_y, int((max_y-min_y)//grid_size+1)))
            x_occ = ((xyz[:,0] - min_x) // grid_size).astype(int)
            y_occ = ((xyz[:,1] - min_y) // grid_size).astype(int)
            grid_empty = np.ones((gridx.shape[0], gridx.shape[1]), dtype=bool)
            grid_empty[y_occ, x_occ] = False

            add_x = gridx[grid_empty] / self.config.scene_scale
            add_y = gridy[grid_empty] / self.config.scene_scale
            add_z = np.ones_like(add_x) * bottom_z
            add_xyz = np.vstack([add_x, add_y, add_z]).T

            add_rgb = np.ones_like(add_xyz) * np.mean(pc.colors, axis=0, keepdims=True)
            add_normal = np.zeros_like(add_xyz)
            add_normal[:, -1] = 1.0

            self.point_cloud = BasicPointCloud(
                points=np.concatenate([pc.points, add_xyz], axis=0),
                normals=np.concatenate([pc.normals, add_normal], axis=0),
                colors=np.concatenate([pc.colors, add_rgb], axis=0)
            )

@dataclass
class DFC2019DataLoaderConfig(SatelliteDataLoaderConfig):
    sun_dist: float = 1e4
    metadata_path: str = ''
    
class DFC2019DataLoader(SatelliteDataLoader):
    '''DataLoader for DFC2019 dataset 
    '''
    config: DFC2019DataLoaderConfig

    def __init__(self, config: DFC2019DataLoaderConfig, source_dir:str, 
                 eval:bool = False, world_size:int = 1, local_rank:int = 0):
        super().__init__(config, source_dir, eval, world_size, local_rank)

        pc = self.point_cloud
        mz, Mz = np.percentile(pc.points[:, -1], 5), np.percentile(pc.points[:, -1], 95)
        mask = (pc.points[:, -1]>mz) & (pc.points[:, -1]<Mz)
        xyz = pc.points[mask]
        xyz *= self.config.scene_scale  # to original scale(meter)
        max_x, min_x = np.max(xyz[:,0]), np.min(xyz[:,0])
        max_y, min_y = np.max(xyz[:,1]), np.min(xyz[:,1])
        scene_bbx = np.array([min_x, min_y, (max_x-min_x), (max_y-min_y)]) / self.config.scene_scale

        for resolution_scale in self.config.resolution_scales:
            train_dataset = self.train_dataset[resolution_scale]
            test_dataset = self.test_dataset[resolution_scale]

            self.train_dataset[resolution_scale] = []
            self.test_dataset[resolution_scale] = []

            for d in train_dataset:
                W2C = d.world_view_transform.T.cpu().numpy()
                points_ic = (W2C[:3,:3] @ self.point_cloud.points.T + W2C[:3, 3:4]).T  #(n,3)
                mean_alt = np.mean(points_ic[:, 2])

                view_index = d.image_name[9:11]
                if config.metadata_path is not None:
                    az, el, sun_az, sun_el = read_imd(os.path.join(config.metadata_path, f'{view_index}.IMD'))
                    sun_view = sun_angles_to_enu(sun_az, sun_el)
                else:
                    sun_view = None

                self.train_dataset[resolution_scale].append(
                    SatelliteCamera(
                        colmap_id=d.colmap_id, 
                        R=d.R, T=d.T, FoVx=d.FoVx, FoVy=d.FoVy,
                        image=d.original_image, gt_alpha_mask=d.gt_alpha_mask,
                        image_name=d.image_name, uid=d.uid,
                        mean_alt=mean_alt,
                        scene_bbx=scene_bbx,
                        sun_view=sun_view,
                        sun_dist=self.config.sun_dist,
                        resolution_scale=d.resolution_scale, 
                        zfar=d.zfar, znear=d.znear,
                        trans=d.trans, scale=d.scale,
                        data_device=d.data_device
                    )
                )

            for d in test_dataset:
                W2C = d.world_view_transform.T.cpu().numpy()
                points_ic = (W2C[:3,:3] @ self.point_cloud.points.T + W2C[:3, 3:4]).T  #(n,3)
                mean_alt = np.mean(points_ic[:, 2])

                view_index = d.image_name[9:11]
                if config.metadata_path is not None:
                    az, el, sun_az, sun_el = read_imd(os.path.join(config.metadata_path, f'{view_index}.IMD'))
                    sun_view = sun_angles_to_enu(sun_az, sun_el)
                else:
                    sun_view = None

                self.test_dataset[resolution_scale].append(
                    SatelliteCamera(
                        colmap_id=d.colmap_id, 
                        R=d.R, T=d.T, FoVx=d.FoVx, FoVy=d.FoVy,
                        image=d.original_image, gt_alpha_mask=d.gt_alpha_mask,
                        image_name=d.image_name, uid=d.uid,
                        mean_alt=mean_alt,
                        scene_bbx=scene_bbx,
                        sun_view=sun_view,
                        sun_dist=self.config.sun_dist,
                        resolution_scale=d.resolution_scale, 
                        zfar=d.zfar, znear=d.znear,
                        trans=d.trans, scale=d.scale,
                        data_device=d.data_device
                    )
                )
        
        # view selection for PGSR
        view_sel = []
        for id in range(len(self.train_dataset[1.0])):
            view_sel.append([(i, 1.0) for i in range(len(self.train_dataset[1.0]))])
        write_pairs(os.path.join(self.source_dir, 'pair.txt'), view_sel)

        for resolution_scale in self.config.resolution_scales:
            for i, cam in enumerate(self.train_dataset[resolution_scale]):
                cam.near_ids = [k for k, s in view_sel[i]]
