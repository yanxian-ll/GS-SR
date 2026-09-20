from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from gssr.pointcloud import BasicPointCloud
from gssr.utils.colmap_loader import (
    Camera,
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
)


@dataclass(frozen=True)
class DepthPriorStats:
    depth_dir: str
    num_frames: int
    num_valid_points: int
    num_sampled_points: int


def _resolve_depth_dir(source_dir: str, depth_dir: str) -> Path:
    path = Path(depth_dir).expanduser()
    if not path.is_absolute():
        path = Path(source_dir) / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Depth prior directory does not exist: {path}")
    return path


def _load_manifest(depth_dir: Path) -> dict:
    path = depth_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Murre manifest: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "predicted":
        raise ValueError(
            f"Murre depth is not ready: manifest status={manifest.get('status')!r}"
        )
    if manifest.get("depth_type") != "camera_z":
        raise ValueError(
            "Depth-prior initialization requires camera_z depth, "
            f"got {manifest.get('depth_type')!r}"
        )
    if not manifest.get("frames"):
        raise ValueError(f"No frames found in Murre manifest: {path}")
    return manifest


def _read_intrinsics_text_relaxed(path: Path) -> Dict[int, Camera]:
    """Read the undistorted COLMAP camera models supported by this project.

    The repository's generic text reader historically asserts PINHOLE only, while
    the normal COLMAP dataloader also supports SIMPLE_PINHOLE. Keep this helper
    local to the depth-prior path so text and binary models behave consistently.
    """
    cameras: Dict[int, Camera] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            if model not in ("PINHOLE", "SIMPLE_PINHOLE"):
                raise ValueError(
                    "Depth-prior initialization requires undistorted "
                    f"PINHOLE/SIMPLE_PINHOLE cameras, got {model}"
                )
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=int(elems[2]),
                height=int(elems[3]),
                params=np.asarray(tuple(map(float, elems[4:])), dtype=np.float64),
            )
    return cameras


def _load_colmap_model(source_dir: str):
    model_dir = Path(source_dir) / "sparse" / "0"
    images_bin = model_dir / "images.bin"
    cameras_bin = model_dir / "cameras.bin"
    images_txt = model_dir / "images.txt"
    cameras_txt = model_dir / "cameras.txt"

    if images_bin.is_file() and cameras_bin.is_file():
        images = read_extrinsics_binary(str(images_bin))
        cameras = read_intrinsics_binary(str(cameras_bin))
    elif images_txt.is_file() and cameras_txt.is_file():
        images = read_extrinsics_text(str(images_txt))
        cameras = _read_intrinsics_text_relaxed(cameras_txt)
    else:
        raise FileNotFoundError(
            f"Missing COLMAP images/cameras model under {model_dir}"
        )

    images_by_stem = {}
    for image in images.values():
        stem = Path(image.name).stem
        if stem in images_by_stem:
            raise ValueError(f"Duplicate COLMAP image stem: {stem}")
        images_by_stem[stem] = image
    return cameras, images_by_stem


def _camera_intrinsic_for_depth(
    camera: Camera,
    processing_res: int,
    depth_shape: Tuple[int, int],
) -> np.ndarray:
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = map(float, camera.params[:4])
    elif camera.model == "SIMPLE_PINHOLE":
        fx, cx, cy = map(float, camera.params[:3])
        fy = fx
    else:
        raise ValueError(
            "Depth-prior initialization requires undistorted "
            f"PINHOLE/SIMPLE_PINHOLE cameras, got {camera.model}"
        )

    scale = (
        float(processing_res) / float(max(camera.width, camera.height))
        if processing_res > 0
        else 1.0
    )
    expected_h = int(camera.height * scale)
    expected_w = int(camera.width * scale)
    # Murre resizes first, then crops only the right/bottom edges to multiples of 8.
    expected_h -= expected_h % 8
    expected_w -= expected_w % 8
    if depth_shape != (expected_h, expected_w):
        raise ValueError(
            "Murre depth shape does not match COLMAP camera + manifest processing_res: "
            f"depth={depth_shape}, expected={(expected_h, expected_w)}, "
            f"camera={camera.width}x{camera.height}, processing_res={processing_res}"
        )

    intrinsic = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsic[:2] *= scale
    return intrinsic


def _world_to_camera(image) -> np.ndarray:
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = qvec2rotmat(image.qvec)
    w2c[:3, 3] = np.asarray(image.tvec, dtype=np.float64)
    return w2c


def _load_depth(depth_dir: Path, frame: dict) -> np.ndarray:
    image_name = frame.get("image")
    if not image_name:
        raise ValueError("Murre manifest frame is missing 'image'")
    stem = Path(image_name).stem
    depth_path = depth_dir / (frame.get("depth") or f"{stem}.npy")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing Murre depth: {depth_path}")

    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"{depth_path}: expected HxW depth, got shape {depth.shape}")
    depth = np.asarray(depth, dtype=np.float32)

    expected_hw = frame.get("depth_hw")
    if expected_hw is not None and tuple(depth.shape) != tuple(expected_hw):
        raise ValueError(
            f"{depth_path}: depth shape {depth.shape} does not match manifest "
            f"{tuple(expected_hw)}"
        )
    return depth


def _frame_colmap_geometry(
    frame: dict,
    depth: np.ndarray,
    processing_res: int,
    cameras: dict,
    images_by_stem: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    image_name = frame["image"]
    stem = Path(image_name).stem
    if stem not in images_by_stem:
        raise KeyError(f"COLMAP sparse/0 has no registered image for depth frame: {image_name}")

    image = images_by_stem[stem]
    if image.camera_id not in cameras:
        raise KeyError(
            f"COLMAP camera {image.camera_id} referenced by {image.name} does not exist"
        )
    camera = cameras[image.camera_id]

    original_hw = frame.get("original_hw")
    if original_hw is not None and tuple(original_hw) != (camera.height, camera.width):
        raise ValueError(
            f"{image_name}: manifest original_hw={tuple(original_hw)} but COLMAP camera is "
            f"{camera.height}x{camera.width}"
        )

    intrinsic = _camera_intrinsic_for_depth(
        camera, processing_res, tuple(depth.shape)
    )
    return intrinsic, _world_to_camera(image)


def _valid_depth_mask(
    depth: np.ndarray,
    depth_min: float,
    depth_max: Optional[float],
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > depth_min)
    if depth_max is not None:
        valid &= depth < depth_max
    return valid


def _backproject_valid_depth(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    w2c: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64, copy=False)

    x = (xs.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (ys.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    xyz_camera = np.stack((x, y, z), axis=1)

    # COLMAP convention: x_camera = R * x_world + t.
    # For row-vector batches: x_world = (x_camera - t) @ R.
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    return (xyz_camera - translation[None, :]) @ rotation


def load_murre_depth_point_cloud(
    source_dir: str,
    depth_dir: str,
    train_image_names: Sequence[str],
    num_points: int,
    seed: int,
    *,
    scale_scene: bool,
    scene_scale: float,
    scene_translation: Sequence[float],
    depth_min: float = 1e-6,
    depth_max: Optional[float] = None,
) -> Tuple[BasicPointCloud, DepthPriorStats]:
    """Build Scaffold initialization points from Murre ``*.npy`` depth.

    Murre contributes only camera-Z depth. Intrinsics and poses are always read
    from the scene's current COLMAP ``sparse/0`` model. This keeps the prior
    initialization tied to exactly the same cameras used by GS-SR and avoids a
    dependency on Murre's temporary ``guidance`` files.

    Only training views are used. Valid depth pixels from all training views form
    one global population, from which ``num_points`` are sampled deterministically.
    Returned points are transformed into the same scaled scene coordinates as the
    existing ``ColmapDataLoader`` point cloud.
    """
    if depth_min < 0:
        raise ValueError(f"depth_min must be non-negative, got {depth_min}")
    if scale_scene and (not np.isfinite(scene_scale) or scene_scale <= 0):
        raise ValueError(f"scene_scale must be positive, got {scene_scale}")

    root = _resolve_depth_dir(source_dir, depth_dir)
    manifest = _load_manifest(root)
    processing_res = int(manifest.get("processing_res", 0))
    if processing_res < 0:
        raise ValueError(f"Invalid Murre processing_res: {processing_res}")

    if depth_max is None:
        manifest_max = manifest.get("max_depth")
        if manifest_max is not None:
            depth_max = float(manifest_max)
    if depth_max is not None and (
        not np.isfinite(depth_max) or depth_max <= depth_min
    ):
        raise ValueError(
            f"depth_max must be finite and greater than depth_min, got {depth_max}"
        )

    train_stems = {Path(name).stem for name in train_image_names}
    if not train_stems:
        raise ValueError("Training split is empty; cannot initialize from depth prior")

    frames_by_stem = {}
    for frame in manifest["frames"]:
        image_name = frame.get("image")
        if not image_name:
            raise ValueError("Murre manifest frame is missing 'image'")
        stem = Path(image_name).stem
        if stem in frames_by_stem:
            raise ValueError(f"Duplicate image stem in Murre manifest: {stem}")
        frames_by_stem[stem] = frame

    missing_depth = sorted(train_stems - frames_by_stem.keys())
    if missing_depth:
        preview = ", ".join(missing_depth[:8])
        suffix = " ..." if len(missing_depth) > 8 else ""
        raise FileNotFoundError(
            f"Murre depth is missing {len(missing_depth)} training views: "
            f"{preview}{suffix}"
        )

    cameras, images_by_stem = _load_colmap_model(source_dir)
    missing_colmap = sorted(train_stems - images_by_stem.keys())
    if missing_colmap:
        preview = ", ".join(missing_colmap[:8])
        suffix = " ..." if len(missing_colmap) > 8 else ""
        raise KeyError(
            f"COLMAP sparse/0 is missing {len(missing_colmap)} training views: "
            f"{preview}{suffix}"
        )

    selected_frames = [
        frame
        for frame in manifest["frames"]
        if Path(frame["image"]).stem in train_stems
    ]

    frame_meta = []
    total_valid = 0
    for frame in selected_frames:
        depth = _load_depth(root, frame)
        intrinsic, w2c = _frame_colmap_geometry(
            frame, depth, processing_res, cameras, images_by_stem
        )
        valid = _valid_depth_mask(depth, depth_min, depth_max)
        count = int(valid.sum())
        if count == 0:
            raise ValueError(f"{frame['image']}: no valid depth pixels for initialization")
        frame_meta.append((frame, count, intrinsic, w2c))
        total_valid += count

    if total_valid == 0:
        raise ValueError("Murre depth prior contains no valid points")

    target_count = total_valid if num_points <= 0 else min(int(num_points), total_valid)
    if target_count == total_valid:
        sampled_global = np.arange(total_valid, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        sampled_global = np.sort(
            rng.choice(total_valid, size=target_count, replace=False).astype(np.int64)
        )

    sampled_points = []
    offset = 0
    for frame, count, intrinsic, w2c in frame_meta:
        begin = np.searchsorted(sampled_global, offset, side="left")
        end = np.searchsorted(sampled_global, offset + count, side="left")
        if begin == end:
            offset += count
            continue

        depth = _load_depth(root, frame)
        valid = _valid_depth_mask(depth, depth_min, depth_max)
        xyz_world = _backproject_valid_depth(depth, intrinsic, w2c, valid)
        local_indices = sampled_global[begin:end] - offset
        sampled_points.append(xyz_world[local_indices])
        offset += count

    points = np.concatenate(sampled_points, axis=0).astype(np.float32, copy=False)
    if scale_scene:
        scene_translation = np.asarray(scene_translation, dtype=np.float64).reshape(1, 3)
        points = (
            (points.astype(np.float64) - scene_translation) / float(scene_scale)
        ).astype(np.float32)

    if points.shape[0] != target_count:
        raise RuntimeError(
            f"Depth-prior sampler produced {points.shape[0]} points, expected {target_count}"
        )

    # ScaffoldGaussian.create_from_data currently consumes only pcd.points. Keep
    # structurally valid zero color/normal arrays without introducing RGB I/O.
    colors = np.zeros_like(points, dtype=np.float32)
    normals = np.zeros_like(points, dtype=np.float32)
    point_cloud = BasicPointCloud(points=points, colors=colors, normals=normals)
    stats = DepthPriorStats(
        depth_dir=str(root),
        num_frames=len(selected_frames),
        num_valid_points=total_valid,
        num_sampled_points=target_count,
    )
    return point_cloud, stats
