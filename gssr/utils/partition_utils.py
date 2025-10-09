import numpy as np
from itertools import compress
from tqdm import tqdm
import os
import shutil
from typing import Tuple, Optional, Dict, List
from scipy.spatial import ConvexHull

from gssr.utils.colmap_read_write_model import read_model, write_model, qvec2rotmat

# from colmap_read_write_model import read_model, write_model, qvec2rotmat


def get_axis_aligned_bounding_box(points:np.ndarray):
    """given a set of points, calculate the axis aligned bounding box. 
    
    Parameters:
    points: numpy array of point coordinates with shape (n,2) / (n,3)
            where n is the number of points
    Output:
        tuple of corners(4,2), centre(2,)

        3-----2
        |     |
        0-----1

    """
    points = points[:, :2]
    # mina = np.min(points, axis=0)
    # maxa = np.max(points, axis=0)
    mina = np.percentile(points, q=1, axis=0)
    maxa = np.percentile(points, q=99, axis=0)

    diff = (maxa - mina) * 0.5
    center = mina + diff
    corners = np.array([center+[-diff[0],-diff[1]],
                        center+[diff[0],-diff[1]],
                        center+[diff[0],diff[1]],
                        center+[-diff[0],diff[1]]])
    print(f"center: {center}, width: {diff[0]*2}, height: {diff[1]*2}")
    return corners, center


def get_oriented_bounding_box(points:np.ndarray, calcconvexhull=True):
    """ given a set of points, calculate the oriented bounding box. 
    
    Parameters:
    points: numpy array of point coordinates with shape (n,2) / (n,3)
            where n is the number of points
    calcconvexhull: boolean, calculate the convex hull of the 
            points before calculating the bounding box. You typically
            want to do that unless you know you are passing in a convex
            point set
    Output:
        tuple of corners(4,2), centre
    """
    points = points[:, :2]

    if calcconvexhull:
        _ch = ConvexHull(points)
        points = _ch.points[_ch.vertices]

    cov_points = np.cov(points, y=None, rowvar=0, bias = 1)
    v, vect = np.linalg.eig(cov_points)
    tvect = np.transpose(vect)

    # use the inverse of the eigenvectors as a rotation matrix and
    # rotate the points so they align with the x and y axes
    points_rotated = np.dot(points, np.linalg.inv(tvect))
    # get the minimum and maximum x and y 
    mina = np.min(points_rotated, axis=0)
    maxa = np.max(points_rotated, axis=0)
    diff = (maxa - mina) * 0.5
    # the center is just half way between the min and max xy
    center = mina + diff

    # get the corners by subtracting and adding half the bounding boxes height and width to the center
    corners = np.array([center+[-diff[0],-diff[1]],
                        center+[diff[0],-diff[1]],
                        center+[diff[0],diff[1]],
                        center+[-diff[0],diff[1]]])
    # use the the eigenvectors as a rotation matrix and
    # rotate the corners and the center back
    corners = np.dot(corners, tvect)
    center = np.dot(center, tvect)

    return corners, center


def split_points_tile(points:np.ndarray, property:Optional[np.ndarray], 
                      bbx:Optional[np.ndarray], tile_size:int=200, min_points:int=1000, extent_ratio:float=1.5):
    """split point cloud based on bounding box.

    Parameters:
    points: numpy array of point coordinates with shape (n,3)
            where n is the number of points
    property: numpy array of point properties with shape (n, m)
            where m is the number of properties(like rgb, mormals)
    bbx: numpy array of bounding-box with shape (4,2). 
            Right now, only support axis-aligned bounding box
                3-----2
                |     |
                0-----1
    Output:
        list of tiles
    """

    ## if bbx is an axis aligned bounding box
    if bbx is None:
        bbx, _ = get_axis_aligned_bounding_box(points)

    w, h = abs(bbx[0, 0] - bbx[1, 0]), abs(bbx[0, 1] - bbx[3, 1])
    # update tile_size
    if (w % tile_size) > tile_size/2: 
        tile_width = w / (w // tile_size + 1)
    else:
        tile_width = w / (w // tile_size)
        
    if (h % tile_size) > tile_size/2:
        tile_height = h / (h // tile_size + 1)
    else:
        tile_height = h / (h // tile_size)

    # get number of colums
    num_col = int(w // tile_width + 1)
    # get number of rows
    num_row = int(h // tile_height + 1)

    x = ((points[:, 0] - bbx[0, 0]) // tile_width).astype(int)
    y = ((points[:, 1] - bbx[0, 1]) // tile_height).astype(int)

    list_tiles = []
    for rr in range(num_row):
        for cc in range(num_col):
            mask = (x==cc) & (y==rr)
            if np.sum(mask) < min_points:
                continue
            
            tile_bbx = np.array([
                            [bbx[0,0] + cc*tile_width, bbx[0,1] + rr*tile_height],
                            [bbx[0,0] + (cc+1)*tile_width, bbx[0,1] + rr*tile_height],
                            [bbx[0,0] + (cc+1)*tile_width, bbx[0,1] + (rr+1)*tile_height],
                            [bbx[0,0] + cc*tile_width, bbx[0,1] + (rr+1)*tile_height]])

            bbx_center = (tile_bbx[0, :] + tile_bbx[2, :]) / 2.0
            extent_bbx = (tile_bbx - bbx_center) * extent_ratio + bbx_center

            list_tiles.append(
                    {
                        "tile_id": cc + rr * num_col,
                        "xyz": points[mask],
                        "property": property[mask] if property is not None else None,
                        "bbx": extent_bbx,
                    }
                )
    
    for i, tile in enumerate(list_tiles):
        tile['tile_id'] = i

    ##TODO: if bbx is an oriented bounding box
    
    return list_tiles


def split_points_quadtree(points:np.ndarray, property:Optional[np.ndarray], 
                          bbx:Optional[np.ndarray], max_points:int=250):
    """split point cloud based on bounding box.

    Parameters:
    points: numpy array of point coordinates with shape (n,3)
            where n is the number of points
    property: numpy array of point properties with shape (n, m)
            where m is the number of properties(like rgb, mormals)
    bbx: numpy array of bounding-box with shape (4,2). 
            Right now, only support axis-aligned bounding box
                3-----2
                |     |
                0-----1

    Output:
        list of tiles
    """

    ## if bbx is an axis aligned bounding box
    if bbx is None:
        bbx, _ = get_axis_aligned_bounding_box(points)

    list_tiles = []
    def quadtree_split(tile) -> None:
        bbx = tile['bbx']
        mx, my, Mx, My = bbx[0,0], bbx[0,1], bbx[2,0], bbx[2,1]
        split_axis = 0 if (Mx-mx)>(My-my) else 1
        index = np.argsort(tile['xyz'][:, split_axis])
        index1, index2 = index[:int(len(index)//2)], index[int(len(index)//2):]
        split1 = {
                "tile_id": -1,
                "xyz": tile['xyz'][index1],
                "property": tile['property'][index1],
                "bbx": get_axis_aligned_bounding_box(tile['xyz'][index1])[0],
            }
        split2 = {
                "tile_id": -1,
                "xyz": tile['xyz'][index2],
                "property": tile['property'][index2],
                "bbx": get_axis_aligned_bounding_box(tile['xyz'][index2])[0],
            }

        if len(split1['xyz']) < max_points:
            list_tiles.append(split1)
        else:
            quadtree_split(split1)
        if len(split2['xyz']) < max_points:
            list_tiles.append(split2)
        else:
            quadtree_split(split2)
    
    # run split
    quadtree_split({
        "tile_id": -1,
        "xyz": points,
        "property": property,
        "bbx": bbx
    })

    for i, tile in enumerate(list_tiles):
        tile['tile_id'] = i

    ##TODO: if bbx is an oriented bounding box

    return list_tiles


def get_IOU(box0: np.ndarray, box1: np.ndarray):
    """calculate IOU

    Parameters
    box0: numpy array with shape (n,4), (xmin, ymin, xmax, ymax)
    box1: numpy array with shape (n,4)

    return: iou (n,)
    """

    xy_max = np.minimum(box0[:, 2:], box1[:, 2:])  #(n,2)
    xy_min = np.maximum(box0[:, :2], box1[:, :2])  #(n,2)

    wc = np.clip(xy_max[:, 0] - xy_min[:, 0], a_min=0, a_max=np.inf)  #(n,)
    hc = np.clip(xy_max[:, 1] - xy_min[:, 1], a_min=0, a_max=np.inf)  #(n,)
    inter = wc * hc

    area_0 = (box0[:, 2] - box0[:, 0]) * (box0[:, 3] - box0[:, 1])
    area_1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    union = area_0 + area_1 - inter

    return inter / union


def update_tiles(list_tiles:List[Dict], cameras:Dict, images:Dict, points3d:Dict, 
                 type:int=1, iou_threshold:float=0.2, extent_ratio:float=1.2):
    """find images, cameras, points3d based on tiled point-cloud. 
        There are two methods to obtain the corresponding images: 
        1) Direct indexing based on SFM results; 
        2) Projecting the bounding box onto the image for determination.
        To enhance applicability, we chose the second method.

    Parameters:
    list_tiles: List(Dict)
    images: colmap SFM result Dict["id"]("id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids")
    cameras: colmap SFM result Dict["id"]("id", "model", "width", "height", "params")
    points3d: colmap SFm result Dict["id"]("id", "xyz", "rgb", "error", "image_ids", "point2D_idxs")
    type: if type=1, pointcloud-based-partition; 
          if type=2, cameraposition-based-partition;
    iou_threshold: float
    extent_ratio: float, only work for type=2

    Output:
        list of tiles
    """
    list_image_keys = list(images.keys())
    list_image_values = list(images.values())

    points = np.vstack([p.xyz for _, p in points3d.items()])

    K_stack = np.zeros((len(images), 3, 3), dtype=np.float32)
    R_stack = np.zeros((len(images), 3, 3), dtype=np.float32)
    t_stack = np.zeros((len(images), 3, 1), dtype=np.float32)
    box_stack = np.zeros((len(images), 4), dtype=np.float32)

    for i, (idx, image) in enumerate(images.items()):
        qvec = image.qvec
        tvec = image.tvec
        cam = cameras[image.camera_id]
        width, height = cam.width, cam.height

        box_stack[i, :] = np.array([0, 0, width, height])

        if cam.model=="SIMPLE_PINHOLE":
            fx = fy = cam.params[0]
        elif cam.model=="PINHOLE":
            fx = cam.params[0]
            fy = cam.params[1]
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        K_stack[i, :, :] = np.array([
            [fx, 0.0, width/2.0],
            [0.0, fy, height/2.0],
            [0.0, 0.0, 1.0]
        ])

        R_stack[i, :, :] = qvec2rotmat(qvec)
        t_stack[i, :, :] = tvec.reshape(3, 1)

    list_new_tiles = []
    for tile in list_tiles:
        bbx = tile['bbx']  # (4,2)
        xyz = tile['xyz']

        if type==1:
            min_z, max_z = np.percentile(xyz[:, -1], q=5), np.percentile(xyz[:, -1], q=95)

            bbx = np.concatenate([
                np.concatenate([bbx, np.ones((4,1))*min_z], axis=1),
                np.concatenate([bbx, np.ones((4,1))*max_z], axis=1)], axis=0)  #(8, 3)

        elif type==2:
            # extent bbx
            bbx_center = (bbx[0, :] + bbx[2, :]) / 2.0
            extent_bbx = (bbx - bbx_center) * extent_ratio + bbx_center
            mx, Mx, my, My = np.min(extent_bbx[:,0]), np.max(extent_bbx[:,0]), np.min(extent_bbx[:,1]), np.max(extent_bbx[:,1])
            # find images and points in bounding-box
            tile_points_mask = (points[:,0] >= mx) & (points[:,0] <= Mx) & (points[:,1] >= my) & (points[:,1] <= My)
            mz, Mz = np.min(points[tile_points_mask][:,2]), np.max(points[tile_points_mask][:,2])
            bbx = np.concatenate([
                np.concatenate([extent_bbx, np.ones((4,1))*mz], axis=1),
                np.concatenate([extent_bbx, np.ones((4,1))*Mz], axis=1)], axis=0)  #(8, 3)
        
        uvs = K_stack @ (R_stack @ bbx.T + t_stack)  # (n,3,8)
        u = uvs[:,0,:] / uvs[:,-1,:]
        v = uvs[:,1,:] / uvs[:,-1,:]

        box = np.vstack([
            np.min(u, axis=1), np.min(v, axis=1), np.max(u, axis=1), np.max(v, axis=1)
        ]).T  #(n,4)

        iou = get_IOU(box, box_stack)  #(n,)
        image_mask = iou > iou_threshold
        tile_images = dict(zip(compress(list_image_keys, image_mask), 
                               compress(list_image_values, image_mask)))

        points_idx = []
        tile_cameras = {}
        for _, img in tile_images.items():
            points_idx.append(img.point3D_ids[img.point3D_ids!=-1])
            tile_cameras[img.camera_id] = cameras[img.camera_id]
        
        tile_points3d = {}
        for idx in np.unique(np.concatenate(points_idx)):
            tile_points3d[idx] = points3d[idx]

        list_new_tiles.append({
            "images": tile_images,
            "cameras": tile_cameras,
            "points3d": tile_points3d,
            "bbx": tile['bbx'],
            "tile_id": tile['tile_id'],
        })
    
    return list_new_tiles


def write_tiles_as_colmap(list_tiles:List[Dict], source_image_path:str, output_path:str, ext='.bin'):
    for _, tile in enumerate(list_tiles):
        tile_images = tile['images']
        tile_points3d = tile['points3d']
        tile_cameras = tile['cameras']
        bbx = tile['bbx']
        tile_id = tile['tile_id']

        ## write sparse/0/...
        tile_name = 'tile_%04d' % tile_id
        output_tile = os.path.join(output_path, f"{tile_name}/sparse/0")
        os.makedirs(output_tile, exist_ok=True)
        write_model(tile_cameras, tile_images, tile_points3d, path=output_tile, ext=ext)
        # CONSOLE.log(f"write {tile_name}, num-images: {len(tile_images)}, num-points: {len(tile_points3d)}")
        print(f"write {tile_name}, num-images: {len(tile_images)}, num-points: {len(tile_points3d)}")

        ## write box for merge
        with open(os.path.join(output_path, f"{tile_name}/box.txt"), 'w') as f:
            f.write(f"x0 y0 x1 y1 x2 y2 x3 y3\n")
            f.write(f"{bbx[0,0]} {bbx[0,1]} {bbx[1,0]} {bbx[1,1]} {bbx[2,0]} {bbx[2,1]} {bbx[3,0]} {bbx[3,1]}")

        ## copy images
        image_path = os.path.join(output_path, f"{tile_name}/images")
        if os.path.exists(image_path): 
            shutil.rmtree(image_path)

        os.makedirs(image_path, exist_ok=True)
        for _, img in tile_images.items():
            orig_path = os.path.join(source_image_path, img.name)
            new_path = os.path.join(image_path, img.name)
            shutil.copy(orig_path, new_path)


def point_cloud_based_partition(source_path:str, output_path:str, tile_size:int, iou_threshold:float=0.02, ext='.bin'):
    cameras, images, points3d = read_model(path=os.path.join(source_path, "sparse/0"))

    points = np.vstack([p.xyz for _, p in points3d.items()])  #(n,3)
    rgbs = np.vstack([p.rgb for _, p in points3d.items()])
    bbx, _ = get_axis_aligned_bounding_box(points)
    
    list_tiles = split_points_tile(points, property=rgbs, bbx=bbx, tile_size=tile_size, min_points=2000)
    # type=1
    list_tiles = update_tiles(list_tiles, cameras, images, points3d, type=1, iou_threshold=iou_threshold)

    write_tiles_as_colmap(list_tiles, os.path.join(source_path, 'images'), output_path, ext=ext)
    return list_tiles


def camera_position_based_partition(source_path:str, output_path:str, max_num_images:int, iou_threshold:float=0.2, extent_ratio:float=1.2, ext='.bin'):
    cameras, images, points3d = read_model(path=os.path.join(source_path, "sparse/0"))

    def get_cam_center(extr):
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        w2c = np.zeros((4,4))
        w2c[:3, :3] = R
        w2c[:3, -1] = T
        w2c[3, 3] = 1.0
        return np.linalg.inv(w2c)[:3, -1]
    
    cam_centers = np.vstack([get_cam_center(img) for _, img in images.items()])  #(n,3)
    cam_bbx, _ = get_axis_aligned_bounding_box(cam_centers)
    cam_idx = np.array(list(images.keys()))[:, None]

    list_tiles = split_points_quadtree(cam_centers, property=cam_idx, bbx=cam_bbx, max_points=max_num_images)
    # type=2
    list_tiles = update_tiles(list_tiles, cameras, images, points3d, type=2, iou_threshold=iou_threshold, extent_ratio=extent_ratio)

    write_tiles_as_colmap(list_tiles, os.path.join(source_path, 'images'), output_path, ext=ext)
    return list_tiles


if __name__ == "__main__":
    source_path = "/home/csuzhang/disk/aerial_dataset/gauu_scene_lower_campus"
    output_path = "output/gauu_scene_lower_campus"

    # camera_position_based_partition(source_path, output_path, max_num_images=250, iou_threshold=0.03)
    point_cloud_based_partition(source_path, output_path, tile_size=10, iou_threshold=0.01)
    