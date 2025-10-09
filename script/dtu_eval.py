import os
from argparse import ArgumentParser

dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']
dtu_scenes = ['scan24']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--method", default="octree-2dgs")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./output/eval_dtu")
parser.add_argument('--dtu', "-dtu", type=str, default="/home/csuzhang/disk/dtu_dataset/dtu")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(dtu_scenes)

if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", type=str, default="/home/csuzhang/disk/dtu_dataset/dtu_eval")
    args = parser.parse_args()


if not args.skip_training:
    # 2dgs scaffold-2dgs octree-2dgs
    if args.method.endswith('2dgs'):
        common_args = "--scene.dataloader.resolution 2 --scene.depth_ratio 0.0 --scene.lambda_dist 1000 --scene.gaussians.voxel_size 0 --scene.gaussians.appearance_dim 0"

    # pgsr scaffold-pgsr octree-pgsr
    if args.method.endswith('pgsr'):
        common_args = "--scene.dataloader.resolution 2 --scene.patch_size 2 --scene.gaussians.voxel_size 0 --scene.gaussians.appearance_dim 0"

    for scene in dtu_scenes:
        source = os.path.join(args.dtu, scene)
        print(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args}")
        os.system(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args}")


if not args.skip_rendering:
    common_args = "--num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    
    for scene in dtu_scenes:
        config = os.path.join(args.output_path, scene, args.method, 'config.yml')
        source = os.path.join(args.dtu, scene)
        print(f"python extract_mesh.py --load_config {config} {common_args}")
        os.system(f"python extract_mesh.py --load_config {config} {common_args}")


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for scene in dtu_scenes:
        scan_id = scene[4:]
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {os.path.join(args.output_path, scene, args.method, 'train/ours_30000/fuse_post.ply')} " + \
            f"--scan_id {scan_id} --output_dir {script_dir}/tmp-dtu/{args.method}/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        os.system(string)
      
