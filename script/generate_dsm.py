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

from gssr.configs import base_config as cfg
from gssr.scene.base_scene import Scene
from gssr.utils.mesh_utils import GaussianExtractor
from gssr.utils.point_utils import depths_to_points
import gssr.utils.dsmr as dsmr
from gssr.utils.metric_utils import evaluate_single

CONSOLE = Console(width=120)

def eval_load_gaussians(config: cfg.TrainerConfig, scene: Scene) -> Path:
    assert config.load_gaussian_dir is not None
    if config.load_gaussian_step is None:
        CONSOLE.log(f"Loading latest gaussians from {config.load_gaussian_dir}")
        if not os.path.exists(config.load_gaussian_dir):
            CONSOLE.rule("Error", style="red")
            CONSOLE.print(f"No gaussians directory found at {config.load_gaussian_dir}, ", justify="center")
            CONSOLE.print(
                "Please make sure the gaussians exists, they should be generated periodically during training",
                justify="center",
            )
            sys.exit(1)
        else:
            load_step = max([int(x[x.find("_") + 1 : x.find(".")]) for x in os.listdir(config.load_gaussian_dir) if x.endswith('.ply')])
            config.load_gaussian_step = load_step
    else:
        load_step = config.load_gaussian_step
    
    load_path = config.load_gaussian_dir / f"iteration_{load_step}.ply"
    scene._gaussians.load_gaussians(load_path)
    scene._gaussians.load_mlp_checkpoints(config.load_gaussian_dir)
    CONSOLE.print(f":white_check_mark: Done loading gaussians from {load_path}")
    return load_path

def eval_setup(config_path: Path, data_device: str = "cuda") -> Tuple[cfg.Config, Scene, Path]:
    # load save config
    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    assert isinstance(config, cfg.Config)

    config.trainer.load_gaussian_dir = config.get_gaussian_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup scene (which includes the dataloader and gaussians)
    config.scene.dataloader.device = data_device
    scene = config.scene.setup(source_dir = config.source_path, eval = config.eval, device = device)
    assert isinstance(scene, Scene)

    # load gaussians information
    gaussian_path = eval_load_gaussians(config.trainer, scene)
    return config, scene, gaussian_path

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


from gssr.utils.render_utils import save_img_f32, save_img_u8, save_vis_depth
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


from tqdm import tqdm
from gssr.utils.image_utils import ssim, psnr

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
    export_image(test_dir, test_viewpoint_stack, rgbmaps, depthmaps, normalmaps)
    return train_rgbmaps, train_depthmaps


@dataclass
class MeshExtractor:
    """Load a gaussian-model, extract mesh"""

    # Path to config YAML file.
    load_config: Optional[Path] = None
    skip_train: bool = False
    skip_test: bool = False
    skip_dsm: bool = False
    # num images selected for merge
    num_images: int = 10
    # merge mode (avg med min max)
    mode: str = 'med'

    data_device: str = "cuda"

    def main(self, load_config=None):
        """Main function."""
        config, scene, _ = eval_setup(config_path=load_config if load_config else self.load_config)
        train_cams = scene.dataloader.getTrainData()
        test_cams = scene.dataloader.getTestData()

        ## setup 
        train_dir = os.path.join(config.get_base_dir(), 'train', "ours_{}".format(config.trainer.load_gaussian_step))
        test_dir = os.path.join(config.get_base_dir(), 'test', "ours_{}".format(config.trainer.load_gaussian_step))
        # gaussExtractor = GaussianExtractor(scene.eval_render)

        rgbmaps, depthmaps = reconstruction(scene.eval_render, train_cams, test_cams, train_dir, test_dir)
        train_psnr, train_ssim = evaluate_single(os.path.join(train_dir, 'renders'), os.path.join(train_dir, 'gt'))
        test_psnr, test_ssim = evaluate_single(os.path.join(test_dir, 'renders'), os.path.join(test_dir, 'gt'))

        # if not self.skip_train:
        #     CONSOLE.log("export training images ...")
        #     os.makedirs(train_dir, exist_ok=True)
            # gaussExtractor.reconstruction(train_cams)
            # gaussExtractor.export_image(train_dir)
            # psnr, ssim = evaluate_single(os.path.join(train_dir, 'renders'), os.path.join(train_dir, 'gt'))

        # if (not self.skip_test) and (len(test_cams) > 0):
        #     CONSOLE.log("export rendered testing images ...")
        #     os.makedirs(test_dir, exist_ok=True)
            # gaussExtractor.reconstruction(test_cams)
            # gaussExtractor.export_image(test_dir)
            # psnr, ssim = evaluate_single(os.path.join(test_dir, 'renders'), os.path.join(test_dir, 'gt'))


        if not self.skip_dsm:
            CONSOLE.log("generate dsm ...")
            os.makedirs(train_dir, exist_ok=True)
            # gaussExtractor.reconstruction(train_cams)

            # sort
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
            ply_dir = os.path.join(config.get_base_dir(), 'train', "ours_{}".format(config.trainer.load_gaussian_step), 'ply')
            os.makedirs(ply_dir, exist_ok=True)
            list_xyz = []
            for idx in sorted_index:
                # depthmap = gaussExtractor.depthmaps[idx]
                # rgbmap = gaussExtractor.rgbmaps[idx].permute(1,2,0)
                depthmap = depthmaps[idx]
                rgbmap = rgbmaps[idx].permute(1,2,0)

                cam = train_cams[idx]
                points = depths_to_points(cam, depthmap.cuda()).detach().cpu().numpy() * scene.dataloader.config.scene_scale
                rgbs = (rgbmap.reshape(-1, 3).detach().cpu().numpy() * 255).astype(np.int8)
                write_ply(os.path.join(ply_dir,  '{0:05d}'.format(idx) + ".ply"), points, rgbs)
                list_xyz.append(points)
            
            metadata_file = os.path.join(config.source_path, 'preprocess/enu_DSM.txt')
            gt_dsm_file = os.path.join(config.source_path, "preprocess/enu_DSM.tif")
            gt_mask_path = os.path.join(config.source_path, "preprocess/enu_CLS.tif")

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

            
            if gt_mask_path is not None:
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
            transform = dsmr.compute_shift(gt_dsm_file, pred_dsm_file, scaling=False)
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


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(MeshExtractor).main()

if __name__ == "__main__":
    entrypoint()
