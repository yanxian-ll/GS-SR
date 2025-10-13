from pathlib import Path
import os
import sys
import torch
import tyro
import numpy as np
from rich.console import Console
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from gssr.cameras import OrthoCamera
from gssr.utils.render_utils import save_img_f32, save_img_u8, save_vis_depth
from gssr.utils.graphics_utils import focal2fov
from utils import eval_setup

CONSOLE = Console(width=120)

# -----------------------------
# helpers
# -----------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n

def _safe_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(np.median(x))

def _get_c2w_from_cam(cam) -> np.ndarray:
    """
    获取 c2w（camera-to-world）4x4矩阵。
    """
    # 优先 world_view_transform（常见于 3DGS 相机）
    if hasattr(cam, "world_view_transform") and cam.world_view_transform is not None:
        w2c = cam.world_view_transform.T.detach().cpu().numpy()
        return np.linalg.inv(w2c)

    # 其次 R/T (world->cam)
    if hasattr(cam, "R") and hasattr(cam, "T") and cam.R is not None and cam.T is not None:
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = cam.R.T
        w2c[:3, 3] = cam.T
        return np.linalg.inv(w2c)

    # 退化：仅位置可用时，构造一个 -Z 朝向的近似姿态
    if hasattr(cam, "camera_center") and cam.camera_center is not None:
        C = cam.camera_center.detach().cpu().numpy().reshape(3)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.array([[1, 0, 0],
                                [0, 1, 0],
                                [0, 0, -1]], dtype=np.float32)
        c2w[:3, 3] = C
        return c2w

    raise RuntimeError("Cannot derive c2w from camera.")


def _derive_nadir_c2w_with_xy_from_view(cam) -> np.ndarray:
    """
    由原视角构造俯视（-Z）但 XY 与原视角在地面投影方向一致的 c2w。
    """
    c2w_view = _get_c2w_from_cam(cam)
    R_cw_view = c2w_view[:3, :3]
    C = c2w_view[:3, 3]

    x_world = R_cw_view[:, 0]  # 原相机 X 轴（世界系）
    x_xy = np.array([x_world[0], x_world[1], 0.0], dtype=np.float32)
    x_xy = _normalize(x_xy)
    if np.linalg.norm(x_xy) < 1e-6:
        x_xy = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    z_nadir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    y_xy = np.cross(z_nadir, x_xy)
    y_xy = _normalize(y_xy)

    R_cw_nadir = np.stack([x_xy, y_xy, z_nadir], axis=1)  # 列为轴
    c2w_nadir = np.eye(4, dtype=np.float32)
    c2w_nadir[:3, :3] = R_cw_nadir
    c2w_nadir[:3, 3] = C
    return c2w_nadir


def _ray_dir_world_from_pixel(cam, x: float, y: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定像素坐标 (x,y)，返回世界系下相机中心 C 和该像素的射线方向 d_w（单位向量）。
    """
    H = int(cam.image_height)
    W = int(cam.image_width)
    fx = float(cam.Fx)
    fy = float(cam.Fy)
    cx = getattr(cam, "Cx", None)
    cy = getattr(cam, "Cy", None)
    if cx is None:
        cx = W * 0.5
    if cy is None:
        cy = H * 0.5

    c2w = _get_c2w_from_cam(cam)
    R = c2w[:3, :3]
    C = c2w[:3, 3]

    dx = (x - cx) / fx
    dy = (y - cy) / fy
    dz = 1.0
    d_cam = np.array([dx, dy, dz], dtype=np.float32)
    d_cam = _normalize(d_cam)

    d_world = (R @ d_cam).astype(np.float32)
    d_world = _normalize(d_world)
    return C, d_world


def _intersect_ray_with_plane(C: np.ndarray, d: np.ndarray, z_plane: float, eps: float = 1e-8) -> Optional[np.ndarray]:
    """
    相机中心 C，方向 d，与平面 z=z_plane 的交点。
    返回 None 表示无效（平行或反向）。
    """
    if abs(d[2]) < eps:
        return None
    t = (z_plane - C[2]) / d[2]
    if t <= 0:
        return None
    P = C + t * d
    return P.astype(np.float32)


def _footprint_polygon_on_plane(cam, z_plane: float) -> Optional[np.ndarray]:
    """
    将原图的四个角像素投影到 z=z_plane，得到地面覆盖四边形（按顺时针）。
    """
    H = int(cam.image_height)
    W = int(cam.image_width)
    corners = [(0.0, 0.0),
               (W - 1.0, 0.0),
               (W - 1.0, H - 1.0),
               (0.0, H - 1.0)]

    pts = []
    for (x, y) in corners:
        C, d = _ray_dir_world_from_pixel(cam, x, y)
        P = _intersect_ray_with_plane(C, d, z_plane)
        if P is None:
            return None
        pts.append(P)
    return np.stack(pts, axis=0)  # (4,3)


def _polygon_uv(poly_xyz: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray, z_plane: float) -> np.ndarray:
    """
    将世界系平面多边形投影到以 x_axis/y_axis 为基（位于 z=z_plane）的 2D (u,v) 平面。
    选用 O=[0,0,z_plane] 为原点，则 u=dot(x_axis,P), v=dot(y_axis,P)。
    """
    u = poly_xyz @ x_axis[:3]
    v = poly_xyz @ y_axis[:3]
    return np.stack([u, v], axis=1).astype(np.float32)  # (N,2)


# -----------------------------
# core
# -----------------------------

@dataclass
class OrthoRender:
    """Render per-train-view orthophotos (nadir) with XY aligned to original view; ground extent from 4 corner intersections."""

    load_config: Path = Path("output/tile_0000/3dgs/2025-10-11_190906/config.yml")
    iterations: Optional[int] = None

    # 保留但不使用的参数
    min_points: int = 0
    extent_ratio: float = 1.05

    # GSD（米/像素），不填则自动估计
    gsd: Optional[float] = None
    tile_size: Optional[float] = None
    camera_height: Optional[float] = None

    data_device: str = "cuda"

    @torch.no_grad()
    def calculate_gsd(self, scene):
        train_dataset = scene.dataloader.getTrainData()
        scale = scene.dataloader.config.scene_scale
        points = scene._gaussians._xyz.detach().cpu().numpy()

        list_camera_height, list_image_size, list_camera_focal = [], [], []
        for cam in train_dataset:
            camera_height = cam.camera_center[-1].detach().cpu().item()
            list_camera_height.append(camera_height)
            list_image_size.extend([cam.image_width, cam.image_height])
            list_camera_focal.extend([cam.Fx, cam.Fy])

        mean_camera_height = sum(list_camera_height) / len(list_camera_height)
        mean_image_size = int(sum(list_image_size) / len(list_image_size))
        mean_camera_focal = int(sum(list_camera_focal) / len(list_camera_focal))

        h_world = mean_camera_height - float(np.mean(points[:, -1]))
        tile_size_world = h_world / mean_camera_focal * mean_image_size
        gsd_world = h_world / mean_camera_focal

        tile_size_m = tile_size_world * scale
        gsd_m = gsd_world * scale
        mean_camera_height_m = mean_camera_height * scale
        return gsd_m, tile_size_m, mean_camera_height_m

    @torch.no_grad()
    def _render_perspective_depth(self, scene, cam) -> Optional[np.ndarray]:
        render_pkg = scene.render(cam)
        if render_pkg is None or ('depth' not in render_pkg):
            return None
        dep = render_pkg['depth']
        if isinstance(dep, torch.Tensor):
            dep = dep.detach().cpu().numpy()
        dep = np.asarray(dep)
        if dep.ndim == 3:
            dep = dep[0]
        return dep

    @torch.no_grad()
    def _estimate_ground_plane_z(self, depth, cam) -> float:
        """
        用透视深度反投影得到地面点的 z 中位数作为 z_plane（更稳健）。
        """
        H = int(cam.image_height)
        W = int(cam.image_width)
        fx = float(cam.Fx)
        fy = float(cam.Fy)
        cx = getattr(cam, "Cx", W * 0.5)
        cy = getattr(cam, "Cy", H * 0.5)

        c2w = _get_c2w_from_cam(cam)
        R = c2w[:3, :3]
        t = c2w[:3, 3]

        us = np.arange(W, dtype=np.float32)
        vs = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs)
        Z = depth.astype(np.float32)
        valid = np.isfinite(Z) & (Z > 0)

        Xc = (uu - cx) / fx * Z
        Yc = (vv - cy) / fy * Z
        Pc = np.stack([Xc, Yc, Z], axis=-1)[valid]
        Pw = (Pc @ R.T) + t
        z_vals = Pw[:, 2]
        z_low = np.percentile(z_vals[np.isfinite(z_vals)], 10)
        G = Pw[z_vals <= (z_low + 1.0)]
        if G.shape[0] < 50:
            G = Pw
        return float(np.median(G[:, 2]))

    @torch.no_grad()
    def _build_ortho_camera_for_view(self, cam, depth, gsd_world: float) -> OrthoCamera:
        """
        为单个训练视角构造正射相机：
        1) 估计地面平面 z_plane；
        2) 计算原图四角在地面的交点多边形；
        3) 在 (u,v) 基上取该四边形的 min/max 作为 ground_width/ground_height 估计；
        4) 由地面宽高 + gsd_world 得到分辨率；
        5) 构造 OrthoCamera（位置同原视角，朝向 -Z，XY 与原视角一致）。
        """
        # (1) 构造 nadir 基
        c2w_nadir = _derive_nadir_c2w_with_xy_from_view(cam)
        x_axis = c2w_nadir[:3, 0]  # XY 平面
        y_axis = c2w_nadir[:3, 1]

        # (2) 地面平面估计
        z_pl = self._estimate_ground_plane_z(depth, cam)

        # (3) 四角→地面四边形
        poly_xyz = _footprint_polygon_on_plane(cam, z_pl)
        if poly_xyz is None:
            # 退化：以相机正下方 2x2 的小范围代替，避免崩溃
            C = _get_c2w_from_cam(cam)[:3, 3]
            poly_xyz = np.array([
                C + np.array([-1, -1, z_pl - C[2]], dtype=np.float32),
                C + np.array([ 1, -1, z_pl - C[2]], dtype=np.float32),
                C + np.array([ 1,  1, z_pl - C[2]], dtype=np.float32),
                C + np.array([-1,  1, z_pl - C[2]], dtype=np.float32),
            ], dtype=np.float32)

        # (4) 多边形→(u,v)，并取轴对齐包围盒估计 ground 宽高
        poly_uv = _polygon_uv(poly_xyz, x_axis, y_axis, z_pl)
        u_min = float(np.min(poly_uv[:, 0]))
        u_max = float(np.max(poly_uv[:, 0]))
        v_min = float(np.min(poly_uv[:, 1]))
        v_max = float(np.max(poly_uv[:, 1]))
        width_u = max(0.0, u_max - u_min)
        height_v = max(0.0, v_max - v_min)
        if width_u <= 1e-6 or height_v <= 1e-6:
            raise RuntimeError("Degenerate ground extent from 4-corner intersections.")

        # (5) 分辨率：像素 = 地面长度 / gsd_world
        img_w = int(np.clip(np.round(width_u / gsd_world), 1, 1e9))
        img_h = int(np.clip(np.round(height_v / gsd_world), 1, 1e9))

        # 相机中心到地面的垂直距离（世界单位）
        Cw = c2w_nadir[:3, 3]
        cam_h_world = float(Cw[2] - z_pl)
        cam_h_world = max(cam_h_world, 1e-4)

        # 等效正射“焦距”
        fx = cam_h_world / gsd_world
        fy = cam_h_world / gsd_world
        FovX = focal2fov(fx, img_w)
        FovY = focal2fov(fy, img_h)

        # w2c / R/T
        w2c = np.linalg.inv(c2w_nadir)
        R = w2c[:3, :3].T
        T = w2c[:3, -1]

        gt_image = torch.ones((3, img_h, img_w), dtype=torch.float32)
        image_name = getattr(cam, "image_name", f"train_{getattr(cam, 'colmap_id', 0)}")
        camera_id = int(getattr(cam, "colmap_id", 50))

        # 以 (u,v) 矩形的 4 个角点还原回世界系坐标（位于 z=z_pl），用于元数据/可视化
        O_plane = np.array([0.0, 0.0, z_pl], dtype=np.float32)
        p00 = O_plane + u_min * x_axis + v_min * y_axis
        p10 = O_plane + u_max * x_axis + v_min * y_axis
        p11 = O_plane + u_max * x_axis + v_max * y_axis
        p01 = O_plane + u_min * x_axis + v_max * y_axis
        rect_corners = np.stack([p00, p10, p11, p01], axis=0)  # (4,3)

        # 世界轴向 AABB（不参与渲染，仅保存）
        xmin = float(np.min(rect_corners[:, 0]))
        xmax = float(np.max(rect_corners[:, 0]))
        ymin = float(np.min(rect_corners[:, 1]))
        ymax = float(np.max(rect_corners[:, 1]))

        ortho_cam = OrthoCamera(
            colmap_id=camera_id,
            R=R,
            T=T,
            FoVx=FovX,
            FoVy=FovY,
            ground_width=width_u,
            ground_height=height_v,
            image=gt_image,
            gt_alpha_mask=None,
            image_name=image_name,
            uid=camera_id,
            data_device=self.data_device,
        )
        setattr(ortho_cam, "bbx", [xmin, xmax, ymin, ymax])
        setattr(ortho_cam, "camera_height", cam_h_world)
        return ortho_cam

    @torch.no_grad()
    def main(self, load_config=None, save=True) -> Tuple[Optional[list], Optional[list], Optional[list], Optional[tuple]]:
        config, scene, _ = eval_setup(
            config_path=load_config if load_config else self.load_config,
            iterations=self.iterations,
            data_device=self.data_device,
        )

        scale = scene.dataloader.config.scene_scale
        tx = scene.dataloader.config.t_x
        ty = scene.dataloader.config.t_y
        tz = scene.dataloader.config.t_z

        # 估计/确认 gsd（米）
        gsd_m, tile_size_m, camera_height_m = self.calculate_gsd(scene)
        if self.gsd is None:
            self.gsd = gsd_m
        if self.tile_size is None:
            self.tile_size = tile_size_m
        if self.camera_height is None:
            self.camera_height = camera_height_m

        gsd_world = self.gsd / scale

        CONSOLE.print(f"[green]Estimated[/green] GSD(m): {self.gsd:.6f}, Tile_size(m): {self.tile_size:.3f}, MeanCamH(m): {self.camera_height:.3f}")
        CONSOLE.print(f"[green]Per-view orthophoto[/green] with gsd_world={gsd_world:.6f} (XY aligned to each view; ground extent from 4-corner intersections)")

        train_dataset = scene.dataloader.getTrainData()
        cameras: List[OrthoCamera] = []
        list_render: List[np.ndarray] = []
        list_ortho_depth: List[Optional[np.ndarray]] = []  # 与正射图对应

        # 先渲染每个训练视角的透视深度，用于估计地面平面
        persp_depths: List[Optional[np.ndarray]] = []
        for idx, train_cam in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="render depth"):
            render_pkg = scene.eval_render(train_cam)
            d = render_pkg['depth'].detach().cpu().numpy() if (render_pkg is not None and 'depth' in render_pkg) else None
            persp_depths.append(d)

        # 每个 train 视角：构建正射相机（四角投影范围）并渲染
        for idx, train_cam in tqdm(enumerate(train_dataset), total=len(train_dataset), desc="per-view orthophoto"):
            try:
                depth = persp_depths[idx]
                if depth is None:
                    raise RuntimeError("No perspective depth for ground-plane estimation.")

                ortho_cam = self._build_ortho_camera_for_view(train_cam, depth, gsd_world)
                cameras.append(ortho_cam)

                # 正射渲染
                means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp, _ = scene.generate_ortho_gaussians(ortho_cam)
                render_pkg = scene.render_ortho(ortho_cam, means3D, opacity, scales, rotations, cov3D_precomp, shs, colors_precomp)

                list_render.append(render_pkg['render'].permute(1, 2, 0).detach().cpu().numpy())
                if 'depth' in render_pkg:
                    list_ortho_depth.append(render_pkg['depth'][0].detach().cpu().numpy())
                else:
                    list_ortho_depth.append(None)

            except Exception as e:
                CONSOLE.print(f"[red]Skip view {idx} due to error: {e}[/red]")

        # 保存
        if save:
            out_dir = os.path.join(config.get_base_dir(), 'ortho', f"per_view_{config.trainer.load_gaussian_step}")
            render_path = os.path.join(out_dir, "renders")
            vis_path = os.path.join(out_dir, "depths")
            coordinate_path = os.path.join(out_dir, "coordinate")
            os.makedirs(render_path, exist_ok=True)
            os.makedirs(vis_path, exist_ok=True)
            os.makedirs(coordinate_path, exist_ok=True)

            for i, ortho_cam in tqdm(enumerate(cameras), total=len(cameras), desc="save per-view results"):
                if i < len(list_render) and list_render[i] is not None:
                    save_img_u8(list_render[i], os.path.join(render_path, f'{i:05d}.png'))

                if i < len(list_ortho_depth) and list_ortho_depth[i] is not None:
                    depth_world = list_ortho_depth[i]
                    # 正射深度到米：相机高度 - 世界深度（再缩放并加偏置）
                    depth_m = (ortho_cam.camera_height - depth_world) * scale + tz
                    save_img_f32(depth_m, os.path.join(vis_path, f'depth_{i:05d}.tiff'))
                    flat = depth_m.flatten()
                    flat = flat[np.isfinite(flat)]
                    lo, hi = np.percentile(flat, [3, 97])
                    save_vis_depth(depth_m, lo, hi, os.path.join(vis_path, f'depth_vis_{i:05d}.png'))

                # 坐标/元数据文件：W,H,GSD,xmin(m),ymin(m)
                # 这里 xmin/ymin 来源于以世界轴向计算的 AABB 的 minX/minY
                xmin, xmax, ymin, ymax = ortho_cam.bbx
                with open(os.path.join(coordinate_path, f'{i:05d}.txt'), 'w') as f:
                    f.write(f"{ortho_cam.image_width}\n")
                    f.write(f"{ortho_cam.image_height}\n")
                    f.write(f"{self.gsd}\n")
                    f.write(f"{xmin * scale + tx}\n")
                    f.write(f"{ymin * scale + ty}\n")
                    
        return cameras, list_render, list_ortho_depth, (scale, tx, ty, tz)


# -----------------------------
# entry
# -----------------------------

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(OrthoRender).main()


if __name__ == "__main__":
    entrypoint()
