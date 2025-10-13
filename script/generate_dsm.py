from pathlib import Path
import yaml
import os
import sys
import torch
import tyro
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import rasterio
from tqdm import tqdm

from plyflatten import plyflatten
from plyflatten.utils import rasterio_crs, crs_proj
import affine
from pyproj import CRS

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from utils import eval_setup
from gssr.utils.metric_utils import evaluate_single
import gssr.utils.dsmr as dsmr
from gssr.utils.image_utils import ssim, psnr
from gssr.utils.render_utils import save_img_f32, save_img_u8, save_vis_depth

CONSOLE = Console(width=120)

def write_ply(filename, xyzs, rgbs=None, normals=None):
    from plyfile import PlyElement, PlyData

    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if normals is None:
        normals = np.zeros_like(xyzs)
    if rgbs is None:
        rgbs = (np.ones_like(xyzs) * 255).astype(np.uint8)

    elements = np.empty(xyzs.shape[0], dtype=dtype)
    attributes = np.concatenate((xyzs, normals, rgbs), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(filename)

# https://github.com/centreborelli/satnerf/blob/78cabda4ea6e89c0fe09d37fbc149c8ccb706151/eval_s2p.py
def project_cloud_into_grid(xyz, meta, mode):
    # possible modes = 'min', 'max', 'avg', 'med'

    easting, northing, pixels, gsd = meta
    dsm_w, dsm_h = pixels, pixels
    origin = np.array([easting, northing])

    map_np = np.zeros((dsm_h, dsm_w), dtype=float)
    map_np[:, :] = np.nan
    coords = np.round((xyz[:, :2]-origin) / gsd).astype(int)

    # sanity check
    valid_rows = np.logical_and(coords[:, 1] < dsm_h, coords[:, 1] >= 0)
    valid_cols = np.logical_and(coords[:,0] < dsm_w, coords[:,0] >= 0)
    valid_coords_indices = np.logical_and(valid_rows, valid_cols)
    coords = coords[valid_coords_indices, :]
    xyz = xyz[valid_coords_indices, :]

    if mode == 'min' or mode == 'max':
        if mode == 'min':
            idx = np.flip(np.argsort(xyz[:,2]))
        else:
            idx = np.argsort(xyz[:,2])   
        coords, data_np = coords[idx], xyz[idx]
        map_np[coords[:,1], coords[:,0]] = data_np[:,2]
    else:
        coords_unique, coords_indices = np.unique(coords, return_inverse=True, axis=0)
        sorted_id_z = sorted(list(zip(coords_indices, xyz[:,2])), key=lambda x: x[0])
        from itertools import groupby
        groups_id_z = groupby(sorted_id_z, lambda x: x[0])

        dsm_z = []
        if mode == 'avg':
            #dsm_z = [np.mean(cloud_heights[coords_indices == i]) for i in np.arange(coords_unique.shape[0])]
            dsm_z = [np.mean(np.array(list(g))[:,1]) for k, g in groups_id_z]
        else:
            #dsm_z = [np.median(cloud_heights[coords_indices == i]) for i in np.arange(coords_unique.shape[0])] # (~180s/dsm)
            dsm_z = [np.median(np.array(list(g))[:,1]) for k, g in groups_id_z] #(~10s/dsm)
            
        map_np[coords_unique[:,1], coords_unique[:,0]] = np.array(dsm_z)
    
    if np.sum(np.logical_not(np.isnan(map_np))) < 3:
        print ('There are less than 3 points.')
    
    raw_map_np = map_np.copy()
    raw_map_np = np.flipud(raw_map_np)
    return raw_map_np

def export_image(path, viewpoint_stack, rgbmaps, depthmaps, normals):
    render_path = os.path.join(path, "renders")
    gts_path = os.path.join(path, "gt")
    vis_path = os.path.join(path, "vis")
    os.makedirs(render_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    if len(depthmaps) > 0 or len(normals) > 0:
        os.makedirs(vis_path, exist_ok=True)

    for idx, viewpoint_cam in tqdm(enumerate(viewpoint_stack), desc="export images"):
        gt = viewpoint_cam.original_image[0:3, :, :]
        save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        save_img_u8(rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))

        if len(depthmaps) > 0:
            depth_frame = depthmaps[idx][0].cpu().numpy()
            p=3
            distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])
            lo, hi = [np.log(x) for x in distance_limits]

            save_img_f32(depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            save_vis_depth(depthmaps[idx][0].cpu().numpy(), lo, hi, os.path.join(vis_path, 'depth_vis_{0:05d}'.format(idx) + ".png"))
        if len(normals) > 0:
            save_img_u8(normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))

def reconstruction(render, train_viewpoint_stack, test_viewpoint_stack, train_dir, test_dir):
    # recon train
    train_rgbmaps, train_normalmaps, train_depthmaps = [], [], []
    for i, viewpoint_cam in tqdm(enumerate(train_viewpoint_stack), desc="reconstruct radiance fields"):
        render_pkg = render(viewpoint_cam)
        rgb = render_pkg['render']
        train_rgbmaps.append(rgb.cpu())
        if 'normal' in render_pkg:
            normal = torch.nn.functional.normalize(render_pkg['normal'], dim=0)
            train_normalmaps.append(normal.cpu())
        if 'depth' in render_pkg:
            depth = render_pkg['depth']
            train_depthmaps.append(depth.cpu())
    export_image(train_dir, train_viewpoint_stack, train_rgbmaps, train_depthmaps, train_normalmaps)

    # recon test
    rgbmaps, normalmaps, depthmaps = [], [], []
    for i, viewpoint_cam in tqdm(enumerate(test_viewpoint_stack), desc="reconstruct radiance fields"):
        psnrs, ssims = [], []
        rgbs, normals, depths = [], [], []
        for cam in train_viewpoint_stack:
            viewpoint_cam.uid = cam.uid
            render_pkg = render(viewpoint_cam)  
            rgb = render_pkg['render']
            gt = viewpoint_cam.original_image[0:3, :, :]
            psnrs.append(psnr(gt, rgb).mean())
            ssims.append(ssim(gt, rgb))
            rgbs.append(rgb.cpu())
            if 'normal' in render_pkg:
                normal = torch.nn.functional.normalize(render_pkg['normal'], dim=0)
                normals.append(normal.cpu())
            if 'depth' in render_pkg:
                depth = render_pkg['depth']
                depths.append(depth.cpu())
        # find best
        idx = torch.argmax(torch.tensor(psnrs))
        rgbmaps.append(rgbs[idx])
        if len(normals) > 0:
            normalmaps.append(normals[idx])
        if len(depths) > 0:
            depthmaps.append(depths[idx])
    if (len(depthmaps) > 0):
        export_image(test_dir, test_viewpoint_stack, rgbmaps, depthmaps, normalmaps)
    return train_rgbmaps, train_depthmaps

def depth_to_world_points(depth_map, view):
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)  # (1, H, W) -> (H, W)
    
    H, W = depth_map.shape
    device = depth_map.device
    dtype = depth_map.dtype
    
    W2C = np.zeros((4, 4))
    W2C[:3, :3] = view.R.transpose()
    W2C[:3, 3] = view.T
    W2C[3, 3] = 1.0
    C2W = torch.tensor(W2C, device=device, dtype=dtype).inverse()

    u = torch.arange(W, device=device, dtype=dtype)
    v = torch.arange(H, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')  # (H, W)

    Z = depth_map  # (H, W)
    X = (u - view.Cx) * Z / view.Fx  # (H, W)
    Y = (v - view.Cy) * Z / view.Fy  # (H, W)
    
    points_cam = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)    
    points_cam_flat = points_cam.reshape(-1, 3)  # (H*W, 3)    
    points_cam_flat_t = torch.concatenate([
        points_cam_flat.t(), \
        torch.ones((1, points_cam_flat.shape[0]), device=device, dtype=dtype)], dim=0)  # (4, N)    
    points_world = C2W @ points_cam_flat_t  # (4, H*W)
    points_world = points_world.t()[:,:3]  # (H*W, 3)
    return points_world

@dataclass
class DSMGenerator:
    """Load a gaussian-model, generate dsm"""
    # Path to config YAML file.
    load_config: Path = Path()
    # num images selected for merge
    num_images: int = -1
    resolution: float = 0.5  # meter
    # merge mode (avg med min max)
    mode: str = 'med'
    data_device: str = "cuda"
    # Whether to compute metrics
    compute_metrics: bool = False
    metadata_file: Optional[Path] = None
    gt_dsm_file: Optional[Path] = None
    gt_mask_path: Optional[Path] = None

    def main(self, load_config=None):
        """Main function."""
        config, scene, _ = eval_setup(config_path=load_config if load_config else self.load_config)
        train_cams = scene.dataloader.getTrainData()
        test_cams = scene.dataloader.getTestData()

        ## setup
        train_dir = os.path.join(config.get_base_dir(), 'train', "ours_{}".format(config.trainer.load_gaussian_step))
        test_dir = os.path.join(config.get_base_dir(), 'test', "ours_{}".format(config.trainer.load_gaussian_step))

        rgbmaps, depthmaps = reconstruction(scene.eval_render, train_cams, test_cams, train_dir, test_dir)
        train_psnr, train_ssim = evaluate_single(os.path.join(train_dir, 'renders'), os.path.join(train_dir, 'gt'))
        if len(test_cams) > 0:
            test_psnr, test_ssim = evaluate_single(os.path.join(test_dir, 'renders'), os.path.join(test_dir, 'gt'))

        if self.num_images < 0:
            self.num_images = len(train_cams)

        # sort according to angle
        list_angles = []
        for cam in train_cams:
            w2c = cam.world_view_transform.T.detach().cpu().numpy()
            c2w = np.linalg.inv(w2c)
            ndir = c2w[:3, 2]
            true_ndir = np.array([0.0, 0.0, -1.0])
            dot_product = np.dot(ndir, true_ndir)
            cos_theta = dot_product / (np.linalg.norm(ndir)*np.linalg.norm(true_ndir))
            angle_rad = np.arccos(cos_theta)
            list_angles.append(np.degrees(angle_rad))
        num_images = self.num_images if self.num_images < len(train_cams) else len(train_cams)
        sorted_index = np.argsort(np.array(list_angles))[:num_images]

        # save as point cloud
        dsm_dir = os.path.join(config.get_base_dir(), 'train', "ours_{}".format(config.trainer.load_gaussian_step), 'dsm')
        os.makedirs(dsm_dir, exist_ok=True)
        list_xyz = []
        for idx in tqdm(sorted_index, desc="export point clouds"):
            depthmap = depthmaps[idx]
            rgbmap = rgbmaps[idx].permute(1,2,0)

            cam = train_cams[idx]
            points = depth_to_world_points(depthmap.cuda(), cam).detach().cpu().numpy()
            points *= scene.dataloader.config.scene_scale
            points[:, 0] += scene.dataloader.config.t_x
            points[:, 1] += scene.dataloader.config.t_y
            points[:, 2] += scene.dataloader.config.t_z
            list_xyz.append(points)
            rgbs = (rgbmap.reshape(-1, 3).detach().cpu().numpy() * 255).astype(np.int8)
            write_ply(os.path.join(dsm_dir, '{0:05d}'.format(idx) + ".ply"), points, rgbs)
        
            # Flatten point clouds
            xmin, xmax = points[:, 0].min(), points[:, 0].max()
            ymin, ymax = points[:, 1].min(), points[:, 1].max()
            xoff = np.floor(xmin / self.resolution) * self.resolution
            xsize = int(1 + np.floor((xmax - xoff) / self.resolution))
            yoff = np.ceil(ymax / self.resolution) * self.resolution
            ysize = int(1 - np.floor((ymin - yoff) / self.resolution))

            with open(os.path.join(dsm_dir, '{0:05d}'.format(idx) + "_enu_bbx.txt"), 'w') as f:
                f.write(f"{xoff}\n{yoff}\n{xsize}\n{ysize}\n{self.resolution}")

            # run plyflatten
            dsm = plyflatten(points, xoff, yoff, self.resolution, xsize, ysize, radius=1, sigma=float("inf"))
            profile = {}
            profile["dtype"] = dsm.dtype
            profile["height"] = dsm.shape[0]
            profile["width"] = dsm.shape[1]
            profile["count"] = 1
            profile["driver"] = "GTiff"
            profile["nodata"] = float("nan")
            profile["crs"] = None
            profile["transform"] = affine.Affine(self.resolution, 0.0, xoff, 0.0, -self.resolution, yoff)

            with rasterio.open(os.path.join(dsm_dir, '{0:05d}'.format(idx) + ".tif"), "w", **profile) as f:
                f.write(dsm[:, :, 0], 1)
        
        # Only compute metrics if enabled and GT paths are provided in config
        if self.compute_metrics and self.gt_dsm_file and self.metadata_file:
            metadata_file = self.metadata_file
            gt_dsm_file = self.gt_dsm_file
            gt_mask_path = self.gt_mask_path

            output_path = os.path.join(config.get_base_dir(), 'metric')
            os.makedirs(output_path, exist_ok=True)
            pred_dsm_file = os.path.join(output_path, "pred_DSM.tif")
            pred_rdsm_file = os.path.join(output_path, "pred_rDSM.tif")
            out_err_file = os.path.join(output_path, "error.tif")
            pred_point_cloud_file = os.path.join(output_path, "pred_DSM.ply")
            metric_file = os.path.join(output_path, 'metric.txt')

            # load metadata
            easting, northing, pixels, gsd = np.loadtxt(metadata_file)
            pixels = int(pixels)

            # generate dsm
            with rasterio.open(gt_dsm_file, "r") as f:
                profile = f.profile
                gt_dsm = f.read()[0, :, :]

            gt_min = np.min(gt_dsm)
            gt_max = np.max(gt_dsm)

            xyz = np.concatenate(list_xyz, axis=0)
            mask = (xyz[:, -1] <= gt_max) & (xyz[:, -1] >= gt_min)

            pred_dsm = project_cloud_into_grid(xyz[mask], [easting, northing, pixels, gsd], mode=self.mode)

            if gt_mask_path is not None and os.path.exists(gt_mask_path):
                with rasterio.open(gt_mask_path, "r") as f:
                    mask = f.read()[0, :, :]
                    water_mask = mask.copy()
                    water_mask[mask != 9] = 0
                    water_mask[mask == 9] = 1
            else:
                water_mask = np.zeros_like(pred_dsm).astype(bool)
            
            with rasterio.open(pred_dsm_file,  "w", **profile) as f:
                pred_dsm[water_mask.astype(bool)] = np.nan
                f.write(pred_dsm, 1)
                
            # register and compute mae
            transform = dsmr.compute_shift(self.gt_dsm_file, pred_dsm_file, scaling=False)
            dsmr.apply_shift(pred_dsm_file, pred_rdsm_file, *transform)
            with rasterio.open(pred_rdsm_file, 'r') as f:
                pred_rdsm = f.read()[0, :, :]
                
            error = pred_rdsm - gt_dsm
            with rasterio.open(out_err_file, 'w', **profile) as dst:
                dst.write(error, 1)
            mae = np.nanmean(abs(error.ravel()))
            print(mae)

            # save metric
            with open(metric_file, 'w') as f:
                f.write(f"MAE: {mae}\n")
                f.write(f"PSNR: {train_psnr}, {test_psnr}\n")
                f.write(f"SSIM: {train_ssim}, {test_ssim}\n")

            # save as ply
            enu_e, enu_n = np.meshgrid(np.linspace(easting, easting + pixels * gsd, pred_dsm.shape[1]),
                                        np.linspace(northing + (pixels - 1) * gsd, northing - 1 * gsd, pred_dsm.shape[0]))
            enu_e = enu_e.reshape((-1))
            enu_n = enu_n.reshape((-1))
            enu_u = pred_dsm.reshape((-1))
            mask = (enu_u <= gt_max) & (enu_u >= gt_min)
            write_ply(pred_point_cloud_file, np.vstack([enu_e[mask], enu_n[mask], enu_u[mask]]).T)
        else:
            CONSOLE.print("[yellow]Skipping metric computation: compute_metrics is False or dsm_metadata not found in config[/yellow]")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(DSMGenerator).main()

if __name__ == "__main__":
    entrypoint()