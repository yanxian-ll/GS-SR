import os
import torch
import shutil
import numpy as np
from tqdm import tqdm
from scipy.spatial import ConvexHull
from plyfile import PlyData, PlyElement
from shapely.geometry import Polygon, box

from gssr.utils.colmap_read_write_model import read_model, write_model, \
                                    qvec2rotmat, rotmat2qvec, \
                                    Image, Point3D
from simple_knn._C import distCUDA2


def transform_colmap(input_model, output_model, P, output_format='.txt'):
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

    points = np.vstack(list_xyz)
    rgbs = np.vstack(list_rgb)
    normals = np.zeros_like(points)
    dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    elements = np.empty(points.shape[0], dtype=dtype_full)
    attributes = np.concatenate((points, normals, rgbs), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(os.path.join(output_model, "points3D.ply"))
    return cameras, images_new, points3D_new

def get_cam_center(extr):
    R = qvec2rotmat(extr.qvec)
    T = np.array(extr.tvec)
    w2c = np.zeros((4,4))
    w2c[:3, :3] = R
    w2c[:3, -1] = T
    w2c[3, 3] = 1.0
    return np.linalg.inv(w2c)[:3, -1]

def camera_position_based_region_division(images, m, n):
    images_new = []
    for i, img in images.items():
        images_new.append({"image": img, "center": get_cam_center(img)})

    images = images_new.copy()

    num_images = len(images)
    num_img_per_col = num_images // m
    sorted_images = sorted(images, key=lambda x: x['center'][0])
    tiles_col_images = []
    for i in range(m):
        tiles_col_images.append(sorted_images[i*num_img_per_col: (
                (i+1)*num_img_per_col if (i+1)*num_img_per_col<num_images else num_images)])
        # print(f"m_tile_{i} info, num-images:{len(tiles_col_images[i])}, \
            #   min:{tiles_col_images[i][0]['center'][0]}, max:{tiles_col_images[i][-1]['center'][0]}")
    
    tiles = []
    for idx, col_images in enumerate(tiles_col_images):
        num_images = len(col_images)
        num_img_per_tile = num_images // n
        sorted_images = sorted(col_images, key=lambda x: x['center'][1])
        for i in range(n):
            tiles.append({"images": sorted_images[i*num_img_per_tile: (
                (i+1)*num_img_per_tile if (i+1)*num_img_per_tile<num_images else num_images)]})
            mx = min([c['center'][0] for c in tiles[-1]['images']])
            Mx = max([c['center'][0] for c in tiles[-1]['images']])
            my = min([c['center'][1] for c in tiles[-1]['images']])
            My = max([c['center'][1] for c in tiles[-1]['images']])
            tiles[-1].update({
                "images": [img['image'] for img in tiles[-1]['images']],
                "box": np.array([mx, Mx, my, My])})
            # print(f"num-images:{len(tiles[-1]['images'])}, mx:{mx}, Mx:{Mx}, my:{my}, My:{My}")
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
        new_tiles.append({'images': images_in_box, 'box': new_box, 'points3D': points3D_in_box})
        # print(f"tile_{i}_info, num-images:{len(new_tiles[i]['images'])}, \
        #     num-points:{len(new_tiles[i]['points3D'])}, \
        #     mx:{new_box[0]}, Mx:{new_box[1]}, my:{new_box[2]}, My:{new_box[3]}")
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
        # mz = min([p.xyz[-1] for p in points3D_tile])
        # Mz = max([p.xyz[-1] for p in points3D_tile])
        new_tiles.append({'box': np.array([mx, Mx, my, My, mz, Mz]), 'points3D': points3D_tile})
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
        new_tiles[i]["images"] = new_images["images"] + tile["images"]
        new_box = new_tiles[i]['box']
        # print(f"tile_{i}_info, num-images:{len(new_tiles[i]['images'])}, num-points:{len(new_tiles[i]['points3D'])}, \
        #     mx:{new_box[0]}, Mx:{new_box[1]}, my:{new_box[2]}, My:{new_box[3]}, mz:{new_box[4]}, Mz:{new_box[5]}")
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
        # print(f"tile_{i}_info, num-images:{len(tile['images'])}, num-points:{len(points)}")
    return new_tiles


# refer-code: https://github.com/kangpeilun/VastGaussian
# parper: VastGaussian: Vast 3D Gaussians for Large Scene Reconstruction
def split_scene(scene_path, output_path, m, n, transform_matrix=None, ratio=0.2, threshold=0.25, input_format="", output_format=".txt"):
    os.makedirs(output_path, exist_ok=True)

    # transform colmap-sfm result
    if transform_matrix is None:
        P = np.identity(4)
    else:
        assert((transform_matrix.shape[0]==4) & (transform_matrix.shape[0]==1) )
        P[:3, -1] = 0   # remove translation    
        # remove scale in rotation matrix
        R = P[:3, :3]
        scales = np.sqrt(np.sum(R*R, axis=1))
        R_new = R / scales[:, np.newaxis]
        P[:3, :3] = R_new
    
    ## transform coordinate, so that the Z-axis of the world coordinate is perpendicular to the ground plane
    cameras, images, points3D = transform_colmap(os.path.join(scene_path, "sparse/0"), 
                                                 os.path.join(output_path, "sparse/aligned"), P, output_format)

    # Camera-position-based region division
    print("\nCamera-position-based region division")
    tiles = camera_position_based_region_division(images, m, n)
    
    # Position-based data selection
    print("\nPosition-based data selection")
    tiles = position_based_data_selection(tiles, images, points3D, ratio=ratio)
    
    # Visibility-based camera selection
    print("\nVisibility-based camera selection")
    tiles = visibility_based_camera_selection(tiles, images, cameras, threshod=threshold)

    # Coverage-based point selection
    print("\nCoverage-based point selection")
    tiles = coverage_based_point_selection(tiles, points3D)

    # write tiles as colmap form
    print("\nWrite tiles as colmap form")
    for i, tile in tqdm(enumerate(tiles)):
        images_tile = tile['images']
        points3D_tile = tile['points3D']
        images_tile_dict, points3D_tile_dict = {}, {}
        for img in images_tile: images_tile_dict[img.id]=img
        for point in points3D_tile: points3D_tile_dict[point.id]=point

        # write sparse/0/...
        tile_name = 'tile_%04d' % i
        print(f"Save {tile_name}:")
        print(f"    num images of: {len(images_tile_dict)}", f"    num points: {len(points3D_tile_dict)}")
        output_tile = os.path.join(output_path, f"{tile_name}/sparse/0")
        os.makedirs(output_tile, exist_ok=True)
        write_model(cameras, images_tile_dict, points3D_tile_dict, path=output_tile, ext=output_format)

        # write box for merge
        box = tile['box']
        with open(os.path.join(output_path, f"{tile_name}/box.txt"), 'w') as f:
            f.write(f"mx Mx my My mz Mz")
            f.write(f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]} {box[5]}")

        # copy images
        image_path = os.path.join(output_path, f"{tile_name}/images")
        if os.path.exists(image_path): shutil.rmtree(image_path)
        os.makedirs(image_path, exist_ok=True)
        for img in images_tile:
            orig_path = os.path.join(scene_path, 'images', img.name)
            new_path = os.path.join(image_path, img.name)
            shutil.copy(orig_path, new_path)
            