#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import logging
import shutil
from argparse import ArgumentParser

# This Python script is based on the shell converter script provided in the MipNerF 360 repository.
parser = ArgumentParser("Colmap converter")
parser.add_argument("--source_path", "-s", required=True, type=str, help="The input images needs to be under the subpath './input'")
parser.add_argument("--max_image_size", default=1600, type=int)
parser.add_argument("--use_docker", action='store_true', help='use docker colmap')
parser.add_argument("--use_aligner", action='store_true', help='use colmap model_orientation_aligner for vastgaussian')
args = parser.parse_args()

if args.use_docker:
    colmap_command = f"docker run -it --rm --gpus all -u $(id -u):$(id -g) \
                        -w /workspace -v {args.source_path}:/workspace \
                        colmap/colmap:latest colmap"
    scene_path = '/workspace'
else:
    colmap_command = "colmap"
    scene_path = args.source_path


os.makedirs(os.path.join(args.source_path, "distorted/sparse"), exist_ok=True)
## Feature extraction
feat_extracton_cmd = f"{colmap_command} feature_extractor \
    --database_path {os.path.join(scene_path, 'distorted/database.db')} \
    --image_path {os.path.join(scene_path, 'input')} \
    --ImageReader.camera_model OPENCV \
    --SiftExtraction.use_gpu {True}"
exit_code = os.system(feat_extracton_cmd)
if exit_code != 0:
    logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
    exit(exit_code)

## Feature matching
feat_matching_cmd = f"{colmap_command} exhaustive_matcher \
            --database_path {os.path.join(scene_path, 'distorted/database.db')} \
            --SiftMatching.use_gpu {True}"
exit_code = os.system(feat_matching_cmd)
if exit_code != 0:
    logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
    exit(exit_code)

### Bundle adjustment
# The default Mapper tolerance is unnecessarily large,
# decreasing it speeds up bundle adjustment steps.
mapper_cmd = f"{colmap_command} mapper \
            --database_path {os.path.join(scene_path, 'distorted/database.db')} \
            --image_path {os.path.join(scene_path, 'input')} \
            --output_path {os.path.join(scene_path, 'distorted/sparse')} \
            --Mapper.ba_global_function_tolerance=0.000001"
exit_code = os.system(mapper_cmd)
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

## orientation_aligner
if args.use_aligner:
    os.makedirs(os.path.join(args.source_path, "distorted/sparse/aligned"), exist_ok=True)
    orientation_aligner = f"{colmap_command} model_orientation_aligner \
        --image_path {os.path.join(scene_path, 'input')} \
        --input_path {os.path.join(scene_path, 'distorted/sparse/0')} \
        --output_path {os.path.join(scene_path, 'distorted/sparse/aligned')}"
    exit_code = os.system(orientation_aligner)
    if exit_code != 0:
        logging.error(f"orientation_aligner failed with code {exit_code}. Exiting.")
        exit(exit_code)
    sparse_path = os.path.join(scene_path, "distorted/sparse/aligned")
else:
    sparse_path = os.path.join(scene_path, "distorted/sparse/0")

### Image undistortion
## We need to undistort our images into ideal pinhole intrinsics.
img_undist_cmd = f"{colmap_command} image_undistorter \
    --image_path {os.path.join(scene_path, 'input')} \
    --input_path {sparse_path} \
    --output_path {scene_path} \
    --output_type COLMAP \
    --min_scale 1.0 \
    --max_image_size {args.max_image_size}"
exit_code = os.system(img_undist_cmd)
if exit_code != 0:
    logging.error(f"image_undistorter failed with code {exit_code}. Exiting.")
    exit(exit_code)

files = os.listdir(os.path.join(args.source_path, "sparse"))
os.makedirs(os.path.join(args.source_path, "sparse/0"), exist_ok=True)
# Copy each file from the source directory to the destination directory
for file in files:
    if file == '0':
        continue
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    shutil.move(source_file, destination_file)

try:
    shutil.rmtree(os.path.join(args.source_path, 'stereo'))
    os.remove(os.path.join(args.source_path, 'run-colmap-geometric.sh'))
    os.remove(os.path.join(args.source_path, 'run-colmap-photometric.sh'))
except:
    pass

print("Done.")
