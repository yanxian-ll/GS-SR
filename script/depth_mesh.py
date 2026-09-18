"""Export a self-contained depth cache, then fuse it without loading GS models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import open3d as o3d


def read_cache(path):
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('version') != 1 or not manifest.get('frames'):
        raise ValueError('无效 depth 缓存')
    for frame in manifest['frames']:
        for key in ('depth', 'rgb'):
            if not (path / frame[key]).is_file():
                raise ValueError('depth 缓存不完整: ' + frame[key])
    return manifest


def export_depth(run, iteration):
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils import eval_setup
    from gssr.utils.mesh_utils import to_cam_open3d, estimate_bounding_sphere
    config, scene, _ = eval_setup(run / 'config.yml', iterations=iteration)
    cams = scene.dataloader.getTrainData()
    calibrated = to_cam_open3d(cams)
    radius, _ = estimate_bounding_sphere(cams)
    target = run / 'depth'
    if target.exists():
        raise ValueError('depth 目录存在但不完整，请备份或移走后重试: ' + str(target))
    stage = Path(tempfile.mkdtemp(prefix='.depth-', dir=str(run)))
    frames = []
    try:
        with torch.no_grad():
            for i, (cam, calib) in enumerate(zip(cams, calibrated)):
                result = scene.eval_render(cam)
                depth = result['depth'].detach().cpu().numpy().reshape(cam.image_height, cam.image_width).copy()
                if cam.gt_alpha_mask is not None:
                    depth[cam.gt_alpha_mask.detach().cpu().numpy().reshape(depth.shape) < .5] = 0
                depth[~np.isfinite(depth) | (depth < 0)] = 0
                rgb = np.ascontiguousarray(np.clip(result['render'].detach().cpu().numpy().transpose(1, 2, 0), 0, 1) * 255, dtype=np.uint8)
                dname, cname = f'{i:05d}.npy', f'{i:05d}_rgb.npy'
                np.save(stage / dname, depth.astype(np.float32), allow_pickle=False)
                np.save(stage / cname, rgb, allow_pickle=False)
                frames.append({'image_name': cam.image_name, 'depth': dname, 'rgb': cname,
                               'intrinsic': calib.intrinsic.intrinsic_matrix.tolist(),
                               'world_to_camera': calib.extrinsic.tolist()})
                print('保存 depth: ' + cam.image_name, flush=True)
        manifest = {'version': 1, 'iteration': iteration, 'source_path': config.source_path,
                    'method': config.method_name, 'depth_type': 'camera_z', 'units': 'model coordinates',
                    'default_depth_trunc': float(radius * 2), 'frames': frames}
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        stage.rename(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def fuse(path, manifest, args):
    depth_trunc = args.depth_trunc if args.depth_trunc > 0 else manifest['default_depth_trunc']
    voxel = args.voxel_size if args.voxel_size > 0 else depth_trunc / args.mesh_res
    trunc = args.sdf_trunc if args.sdf_trunc > 0 else voxel * 5
    if not all(np.isfinite(v) and v > 0 for v in (depth_trunc, voxel, trunc)):
        raise ValueError('无效 TSDF 参数，请显式设置 DEPTH_TRUNC 和 MESH_VOXEL_SIZE')
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel, sdf_trunc=trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for frame in manifest['frames']:
        depth = np.load(path / frame['depth'], allow_pickle=False)
        rgb = np.load(path / frame['rgb'], allow_pickle=False)
        if depth.ndim != 2 or rgb.shape != depth.shape + (3,) or not np.isfinite(depth).all():
            raise ValueError('缓存数据尺寸/数值错误: ' + frame['depth'])
        h, w = depth.shape
        k = np.asarray(frame['intrinsic'])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, k[0, 0], k[1, 1], k[0, 2], k[1, 2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb, dtype=np.uint8)),
            o3d.geometry.Image(np.ascontiguousarray(depth, dtype=np.float32)),
            depth_scale=1., depth_trunc=depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intrinsic, np.array(frame['world_to_camera']))
    mesh = volume.extract_triangle_mesh()
    if not len(mesh.triangles):
        raise ValueError('TSDF 未生成三角面；depth 已保存，可调整融合参数后重试')
    out = args.run_dir / 'mesh'
    out.mkdir(exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(out / 'fuse.ply'), mesh):
        raise IOError('mesh 写入失败')
    labels, counts, _ = mesh.cluster_connected_triangles()
    labels, counts = np.asarray(labels), np.asarray(counts)
    threshold = max(50, sorted(counts, reverse=True)[min(args.num_cluster, len(counts))-1])
    mesh.remove_triangles_by_mask(counts[labels] < threshold)
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    if not o3d.io.write_triangle_mesh(str(out / 'fuse_post.ply'), mesh):
        raise IOError('后处理 mesh 写入失败')
    (out / 'fusion.json').write_text(json.dumps({'depth_trunc': depth_trunc, 'voxel_size': voxel,
        'sdf_trunc': trunc, 'num_cluster': args.num_cluster, 'depth_cache': str(path)}, indent=2)+'\n')
    print('mesh 输出: ' + str(out), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--iteration', type=int, required=True)
    p.add_argument('--source-path', required=True)
    p.add_argument('--method', required=True)
    p.add_argument('--voxel-size', type=float, default=-1)
    p.add_argument('--depth-trunc', type=float, default=-1)
    p.add_argument('--sdf-trunc', type=float, default=-1)
    p.add_argument('--mesh-res', type=int, default=1024)
    p.add_argument('--num-cluster', type=int, default=50)
    a = p.parse_args()
    if a.mesh_res < 1 or a.num_cluster < 1:
        p.error('mesh-res 和 num-cluster 必须为正')
    a.run_dir = a.run_dir.resolve()
    cache = a.run_dir / 'depth'
    m = read_cache(cache) if cache.exists() else export_depth(a.run_dir, a.iteration)
    if m['iteration'] != a.iteration or m['method'] != a.method or Path(m['source_path']).resolve() != Path(a.source_path).resolve():
        raise ValueError('缓存与当前场景/方法/迭代不符，请使用新的 RUN_NAME')
    fuse(cache, m, a)


if __name__ == '__main__':
    main()
