import os
import torch
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, box
from typing import Tuple, Optional, Dict, List
import numpy as np

from gssr.utils.colmap_read_write_model import read_model, write_model, \
                                    qvec2rotmat, rotmat2qvec, \
                                    Image, Point3D
from simple_knn._C import distCUDA2


def transform_colmap(input_model: str, output_model: str, transform_file: str, output_format: str = '.txt') -> Tuple:
    # read transform file
    with open(transform_file, 'r') as f:
        transform_matrix = [np.array(
            [float(item) for item in line.strip().split(" ")]) 
                for line in f.readlines()]
        
    P = np.vstack(transform_matrix)
    assert((P.shape[0]==4) and (P.shape[1]==4)), "transform matrix should be 4x4"

    P[:3, -1] = 0   # remove translation    
    # remove scale in rotation matrix
    R = P[:3, :3]
    scales = np.sqrt(np.sum(R*R, axis=1))
    R_new = R / scales[:, np.newaxis]
    P[:3, :3] = R_new

    cameras, images, points3D = read_model(path=input_model, ext="")
    # print("num_cameras:", len(cameras))
    # print("num_images:", len(images))
    # print("num_points3D:", len(points3D))

    images_new = {}
    for i, extr in images.items():
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        w2c = np.zeros((4,4))
        w2c[:3, :3] = R
        w2c[:3, -1] = T
        w2c[3, 3] = 1.0

        w2c_new = w2c @ np.linalg.inv(P)
        qvec = rotmat2qvec(w2c_new[:3, :3])
        tvec = w2c_new[:3, -1]
        images_new[i] = Image(
            id=extr.id, qvec=qvec, tvec=tvec,
            camera_id=extr.camera_id, name=extr.name,
            xys=extr.xys, point3D_ids=extr.point3D_ids)
    
    points3D_new = {}
    list_xyz, list_rgb = [], []
    for i, point in points3D.items():
        xyz = point.xyz[:, np.newaxis]
        xyz_new = (P[:3, :3] @ xyz + P[:3, 3:4]).flatten()
        points3D_new[i] = Point3D(id=point.id, xyz=xyz_new, 
                              rgb=point.rgb, error=point.error, image_ids=point.image_ids,
                              point2D_idxs=point.point2D_idxs)
        list_xyz.append(xyz_new)
        list_rgb.append(point.rgb)
                              
    os.makedirs(output_model, exist_ok=True)
    write_model(cameras, images_new, points3D_new, path=output_model, ext=output_format)

    # points = np.vstack(list_xyz)
    # rgbs = np.vstack(list_rgb)
    # normals = np.zeros_like(points)
    # dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    # elements = np.empty(points.shape[0], dtype=dtype_full)
    # attributes = np.concatenate((points, normals, rgbs), axis=1)
    # elements[:] = list(map(tuple, attributes))
    # el = PlyElement.describe(elements, 'vertex')
    # PlyData([el]).write(os.path.join(output_model, "points3D.ply"))
    return cameras, images_new, points3D_new

def get_cam_center(extr):
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    w2c = np.zeros((4,4))
    w2c[:3, :3] = R
    w2c[:3, -1] = T
    w2c[3, 3] = 1.0
    return np.linalg.inv(w2c)[:3, -1]


def camera_position_based_region_division(images: Dict, num_col: Optional[int]=None, num_row: Optional[int]=None, max_num_images: int = 150):
    images_new = []
    for i, img in images.items():
        images_new.append({"image": img, "center": get_cam_center(img)})
    images = images_new.copy()

    tiles = []
    list_tiles = []

    if (num_col is None) or (num_row is None):
        def quadtree_split(images: List[Dict]) -> None:
            centers = np.vstack([image['center'] for image in images])
            # get bounding box
            mx, Mx, my, My = np.min(centers[:,0]), np.max(centers[:,0]), np.min(centers[:,1]), np.max(centers[:,1])
            # 
            split_axis = 0 if (Mx-mx)>(My-my) else 1
            # sort by split_axis
            sorted_images = sorted(images, key=lambda x: x['center'][split_axis])
            # split
            split1 = sorted_images[:int(len(sorted_images)//2)]
            split2 = sorted_images[int(len(sorted_images)//2):]
            if len(split1) < max_num_images:
                list_tiles.append(split1)
            else:
                quadtree_split(split1)
            if len(split2) < max_num_images:
                list_tiles.append(split2)
            else:
                quadtree_split(split2)
        # split scene
        quadtree_split(images)
        
    else:
        num_images = len(images)
        num_img_per_col = num_images // num_col
        sorted_images = sorted(images, key=lambda x: x['center'][0])
        col_of_tiles = []
        for i in range(num_col):
            col_of_tiles.append(sorted_images[i*num_img_per_col: (
                    (i+1)*num_img_per_col if (i+1)*num_img_per_col<num_images else num_images)])
    
        for idx, col in enumerate(col_of_tiles):
            num_images = len(col)
            num_img_per_tile = num_images // num_row
            sorted_images = sorted(col, key=lambda x: x['center'][1])
            for i in range(num_row):
                list_tiles.append(sorted_images[i*num_img_per_tile: (
                    (i+1)*num_img_per_tile if (i+1)*num_img_per_tile<num_images else num_images)])
    
    # compute box
    for tile in list_tiles:
        centers = np.vstack([t['center'] for t in tile])
        mx, Mx, my, My = np.min(centers[:,0]), np.max(centers[:,0]), np.min(centers[:,1]), np.max(centers[:,1])
        tiles.append({
            "images": tile,
            "box": np.array([mx, Mx, my, My])
        })

    return tiles


def box_based_update(box, images, points3D):
    images_in_box = []
    for i, img in images.items():
        center = get_cam_center(img)
        if (center[0] >= box[0]) & (center[0] <= box[1]) & (center[1] >= box[2]) & (center[1] <= box[3]):
            # images_in_box.append({"image": img, "center": center,})
            images_in_box.append(img)

    points3D_in_box = []
    for i, point in points3D.items():
        xyz = point.xyz
        if (xyz[0] >= box[0]) & (xyz[0] <= box[1]) & (xyz[1] >= box[2]) & (xyz[1] <= box[3]):
            points3D_in_box.append(point)
    return images_in_box, points3D_in_box

def position_based_data_selection(tiles, images, points3D, ratio=0.2):
    new_tiles = []
    for i, tile in enumerate(tiles):
        mx, Mx, my, My = tile['box']
        dw = (Mx - mx) * ratio / 2.0
        dh = (My - my) * ratio / 2.0
        new_box = np.array([mx-dw, Mx+dw, my-dh, My+dh])
        # update images and points
        images_in_box, points3D_in_box = box_based_update(new_box, images, points3D)
        # new_tiles.append({'images': images_in_box, 'box': new_box, 'points3D': points3D_in_box})

        ## we do not update box
        new_tiles.append({'images': images_in_box, 'box': tile['box'], 'points3D': points3D_in_box})
    return new_tiles


def get_w2c_matrix(extr):
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    w2c = np.zeros((4,4))
    w2c[:3, :3] = R
    w2c[:3, -1] = T
    w2c[3, 3] = 1.0
    return w2c

def get_K_matrix(intr):    
    height = intr.height
    width = intr.width
    if intr.model=="SIMPLE_PINHOLE":
        fx = intr.params[0]
        fy = intr.params[0]
    elif intr.model=="PINHOLE":
        fx = intr.params[0]
        fy = intr.params[1]
    else:
        assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
    K = np.array([[fx, 0.0, width/2.0],[0.0, fy, height/2.0],[0, 0, 1]])
    return K

def projection(points, extr, intr):
    """
    points: (Nx4)
    """
    w2c = get_w2c_matrix(extr)  #(4,4)
    K = get_K_matrix(intr)    #(3,3)
    points_ic = (w2c @ points.T).T  #(N,4)
    points_ic = points_ic[:,:3] / points_ic[:, 3:4]  #(N,3)
    uvs = (K @ points_ic.T).T  #(N,3)
    uvs = uvs[:, :2] / uvs[:, 2:3]  #(N,2)
    return uvs

def visibility_based_camera_selection(tiles, images, cameras, threshod=0.25):
    new_tiles = []
    add_images = []
    for idx, tile in enumerate(tiles):
        current_image_id = [img.id for img in tile['images']]
        # get bounding-box
        mx, Mx, my, My = tile['box']
        points3D_tile = tile['points3D']
        # filter points
        points_tensor = torch.from_numpy(np.array([p.xyz for p in points3D_tile])).float().cuda()
        dist = distCUDA2(points_tensor).float().cpu().numpy()
        mask = (dist>(dist.mean()-3*dist.std())) & (dist<(dist.mean()+3*dist.std()))
        points_valid = points_tensor[mask]
        mz = points_valid[:, -1].min().item()
        Mz = points_valid[:, -1].max().item()
        bounding_box = np.array([[mx, my, mz, 1.0], [mx, my, Mz, 1.0], 
                                 [mx, My, mz, 1.0], [mx, My, Mz, 1.0],
                                 [Mx, my, mz, 1.0], [Mx, my, Mz, 1.0],
                                 [Mx, My, mz, 1.0], [Mx, My, Mz, 1.0],])  #(8x4)
        # compute mean camera-tile distance
        centers = np.vstack([get_cam_center(img) for img in tile['images']])  #(N,3)
        list_d = []
        for p in bounding_box:
            list_d.append(np.sqrt(np.power(centers[:,0]-p[0], 2) + np.power(centers[:,1]-p[1], 2) + np.power(centers[:,2]-p[2], 2)))
        md = np.max(np.vstack(list_d), axis=0).mean() * 1.2

        temp_list = []
        for i, extr in images.items():
            if extr.id not in current_image_id:
                intr = cameras[extr.camera_id]
                uvs = projection(bounding_box, extr, intr)
                # compute intersection area
                convex_hull_idx = ConvexHull(uvs).vertices.tolist()
                convex_hull_uv = uvs[convex_hull_idx]
                poly1 = Polygon(convex_hull_uv)
                poly2 = box(0, 0, intr.width, intr.height)
                intersections = poly1.intersection(poly2)
                ratio = intersections.area / (intr.width*intr.height)
                # compute mean distance
                c = get_cam_center(extr)
                d = np.mean(np.sum(np.sqrt(np.power(bounding_box[:, :3] - c[np.newaxis, :], 2)), axis=1))

                if (ratio > threshod) and (d < md):
                # if (ratio > threshod):
                    temp_list.append(extr)
                    # print(f"adsd new-image {extr.name} into tile_{idx}, ratio:{ratio}, intersection-area:{intersections.area}, image-area:{intr.width*intr.height}")
        add_images.append({"images": temp_list})
    
    # update images
    for i, (new_images, tile) in enumerate(zip(add_images, tiles)):
        new_tiles.append({
            "images": new_images["images"] + tile["images"],
            "box": tile["box"],
            "points3D": tile["points3D"],
        })
    return new_tiles


def coverage_based_point_selection(tiles, points3D):
    new_tiles = []
    for i, tile in enumerate(tiles):
        points = []
        points_idx = []
        for image in tile['images']:
            points_idx.append(image.point3D_ids[image.point3D_ids!=-1])
        points_idx = np.unique(np.concatenate(points_idx))
        for idx in points_idx: points.append(points3D[idx])
        new_tiles.append(
            {"images": tile["images"], "box": tile["box"], "points3D": points}
        )
    return new_tiles
