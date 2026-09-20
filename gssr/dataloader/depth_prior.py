from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from gssr.pointcloud import BasicPointCloud


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
            "Depth-prior initialization currently requires Murre camera_z depth, "
            f"got {manifest.get('depth_type')!r}"
        )
    if manifest.get("pose_convention") != "world_to_camera":
        raise ValueError(
            "Depth-prior initialization currently requires world_to_camera poses, "
            f"got {manifest.get('pose_convention')!r}"
        )
    if not manifest.get("frames"):
        raise ValueError(f"No frames found in Murre manifest: {path}")
    return manifest


def _load_frame_geometry(depth_dir: Path, frame: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    image_name = frame.get("image")
    if not image_name:
        raise ValueError("Murre manifest frame is missing 'image'")

    stem = Path(image_name).stem
    depth_name = frame.get("depth") or f"{stem}.npy"
    depth_path = depth_dir / depth_name
    intrinsic_path = depth_dir / "guidance" / "intrinsic" / f"{stem}.txt"
    pose_path = depth_dir / "guidance" / "pose" / f"{stem}.txt"
    rgb_path = depth_dir / f"{stem}_rgb.png"

    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing Murre depth: {depth_path}")
    if not intrinsic_path.is_file():
        raise FileNotFoundError(f"Missing Murre intrinsic: {intrinsic_path}")
    if not pose_path.is_file():
        raise FileNotFoundError(f"Missing Murre pose: {pose_path}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"Missing Murre aligned RGB: {rgb_path}")

    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError(f"{depth_path}: expected HxW depth, got shape {depth.shape}")
    depth = np.asarray(depth, dtype=np.float32)

    expected_hw = frame.get("depth_hw")
    if expected_hw is not None and tuple(depth.shape) != tuple(expected_hw):
        raise ValueError(
            f"{depth_path}: depth shape {depth.shape} does not match manifest {tuple(expected_hw)}"
        )

    intrinsic = np.loadtxt(intrinsic_path, dtype=np.float64)
    w2c = np.loadtxt(pose_path, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError(f"{intrinsic_path}: invalid 3x3 intrinsic")
    if w2c.shape != (4, 4) or not np.isfinite(w2c).all():
        raise ValueError(f"{pose_path}: invalid 4x4 world_to_camera pose")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError(f"{intrinsic_path}: focal length must be positive")

    return depth, intrinsic, w2c, rgb_path


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
    rgb_path: Path,
    valid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(valid)
    z = depth[ys, xs].astype(np.float64, copy=False)

    x = (xs.astype(np.float64) - intrinsic[0, 2]) * z / intrinsic[0, 0]
    y = (ys.astype(np.float64) - intrinsic[1, 2]) * z / intrinsic[1, 1]
    xyz_camera = np.stack((x, y, z), axis=1)

    # x_camera = R * x_world + t  ->  x_world = R^T * (x_camera - t).
    # For row-vector batches this is (x_camera - t) @ R.
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    xyz_world = (xyz_camera - translation[None, :]) @ rotation

    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"{rgb_path}: RGB shape {rgb.shape[:2]} does not match depth {depth.shape}"
        )
    colors = rgb[ys, xs]
    return xyz_world, colors


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
    """Build an initialization point cloud from Murre ``*.npy`` camera-Z depths.

    Only frames belonging to the training split are used. Sampling is uniform over
    all valid depth pixels across all selected views, and is deterministic for a
    fixed ``seed``. Each selected point is returned in the same scene coordinates
    used by ``ColmapDataLoader``.
    """
    if depth_min < 0:
        raise ValueError(f"depth_min must be non-negative, got {depth_min}")
    if scale_scene:
        if not np.isfinite(scene_scale) or scene_scale <= 0:
            raise ValueError(f"scene_scale must be positive, got {scene_scale}")

    root = _resolve_depth_dir(source_dir, depth_dir)
    manifest = _load_manifest(root)

    if depth_max is None:
        manifest_max = manifest.get("max_depth")
        if manifest_max is not None:
            depth_max = float(manifest_max)
    if depth_max is not None:
        if not np.isfinite(depth_max) or depth_max <= depth_min:
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

    missing = sorted(train_stems - frames_by_stem.keys())
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise FileNotFoundError(
            f"Murre depth is missing {len(missing)} training views: {preview}{suffix}"
        )

    selected_frames = [
        frame for frame in manifest["frames"]
        if Path(frame["image"]).stem in train_stems
    ]

    frame_meta = []
    total_valid = 0
    for frame in selected_frames:
        depth, _, _, _ = _load_frame_geometry(root, frame)
        valid = _valid_depth_mask(depth, depth_min, depth_max)
        count = int(valid.sum())
        if count == 0:
            raise ValueError(f"{frame['image']}: no valid depth pixels for initialization")
        frame_meta.append((frame, count))
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
    sampled_colors = []
    offset = 0
    for frame, count in frame_meta:
        begin = np.searchsorted(sampled_global, offset, side="left")
        end = np.searchsorted(sampled_global, offset + count, side="left")
        if begin == end:
            offset += count
            continue

        depth, intrinsic, w2c, rgb_path = _load_frame_geometry(root, frame)
        valid = _valid_depth_mask(depth, depth_min, depth_max)

        # Back-project the complete valid depth map first, then take this view's
        # portion of the global sample. This avoids materializing all views at once.
        xyz_world, colors = _backproject_valid_depth(
            depth, intrinsic, w2c, rgb_path, valid
        )
        local_indices = sampled_global[begin:end] - offset
        sampled_points.append(xyz_world[local_indices])
        sampled_colors.append(colors[local_indices])
        offset += count

    points = np.concatenate(sampled_points, axis=0).astype(np.float32, copy=False)
    colors = np.concatenate(sampled_colors, axis=0).astype(np.float32, copy=False)

    if scale_scene:
        scene_translation = np.asarray(scene_translation, dtype=np.float64).reshape(1, 3)
        points = (
            (points.astype(np.float64) - scene_translation) / float(scene_scale)
        ).astype(np.float32)
    normals = np.zeros_like(points, dtype=np.float32)

    if points.shape[0] != target_count:
        raise RuntimeError(
            f"Depth-prior sampler produced {points.shape[0]} points, expected {target_count}"
        )

    point_cloud = BasicPointCloud(points=points, colors=colors, normals=normals)
    stats = DepthPriorStats(
        depth_dir=str(root),
        num_frames=len(selected_frames),
        num_valid_points=total_valid,
        num_sampled_points=target_count,
    )
    return point_cloud, stats
