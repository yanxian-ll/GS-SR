import numpy as np
from dataclasses import dataclass, field
from gssr.pointcloud import BasicPointCloud

@dataclass
class SceneInfo():
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    