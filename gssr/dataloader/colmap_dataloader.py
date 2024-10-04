import os
import random
import numpy as np
from dataclasses import dataclass, field
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from gssr.dataloader.utils import getNerfppNorm, storePly, fetchPly
from gssr.utils.graphics_utils import focal2fov
from gssr.utils.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text)
from gssr.cameras import CameraInfo
from gssr.scene import SceneInfo
from gssr.cameras.utils import cameraList_from_camInfos
from gssr.dataloader.base_dataloader import DataLoader, DataLoaderConfig


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        # uid = intr.id
        uid = extr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    return cam_infos

def detect_model_format(path, ext):
    if os.path.isfile(os.path.join(path, "cameras"  + ext)) and \
       os.path.isfile(os.path.join(path, "images"   + ext)) and \
       os.path.isfile(os.path.join(path, "points3D" + ext)):
       print("Detected model format: '" + ext + "'")
       return True
    return False

def readColmapSceneInfo(path, images, eval, llffhold=8):
    if detect_model_format(os.path.join(path, "sparse/0"), ".bin"):
        ext = ".bin"
    elif detect_model_format(os.path.join(path, "sparse/0"), ".txt"):
        ext = ".txt"
    else:
        raise RuntimeError("Only Support '.bin' or '.txt' format")
    
    if ext == ".bin":
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    else:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    if not os.path.exists(ply_path):
        if ext == ".bin":
            point_cloud_file = os.path.join(path, "sparse/0/points3D.bin")
            xyz, rgb, _ = read_points3D_binary(point_cloud_file)
        else:
            point_cloud_file = os.path.join(path, "sparse/0/points3D.txt")
            xyz, rgb, _ = read_points3D_text(point_cloud_file)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


@dataclass
class ColmapDataLoaderConfig(DataLoaderConfig):
    _target: Type = field(default_factory=lambda: ColmapDataLoader)

class ColmapDataLoader(DataLoader):
    config: ColmapDataLoaderConfig
    def __init__(
        self,
        config: ColmapDataLoaderConfig,
        source_dir: str,
        eval: bool = False,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        super().__init__(config, source_dir, eval, world_size, local_rank)

        assert(os.path.exists(os.path.join(self.source_dir, "sparse")), "colmap/sparse not exists")
        scene_info = readColmapSceneInfo(self.source_dir, self.config.images, self.eval, self.config.llffhold)

        if self.config.shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.point_cloud = scene_info.point_cloud

        for resolution_scale in self.config.resolution_scales:
            print("Loading Training Data")
            self.train_dataset[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, self.config.resolution, self.config.device)
            print("Loading Test Data")
            self.test_dataset[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, self.config.resolution, self.config.device)
    