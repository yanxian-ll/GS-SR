import os
from argparse import ArgumentParser

tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
tnt_large_scenes = ['Meetingroom', 'Courthouse']

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--method", default="scaffold-2dgs")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./output/eval_tnt")
parser.add_argument('--TNT_data', "-TNT_data", type=str, default="/home/csuzhang/disk/tnt_dataset/tnt")
args, _ = parser.parse_known_args()

if not args.skip_metrics:
    parser.add_argument('--TNT_GT', type=str, default="/home/csuzhang/disk/tnt_dataset/tnt_eval")
    args = parser.parse_args()


if not args.skip_training:
    # 2dgs scaffold-2dgs octree-2dgs
    if args.method.endswith('2dgs'):
        common_args = "--scene.dataloader.resolution 2 --scene.depth_ratio 0.0 --scene.gaussians.voxel_size 0 --scene.gaussians.appearance_dim 0"

    # pgsr scaffold-pgsr octree-pgsr
    if args.method.endswith('pgsr'):
        common_args = "--scene.dataloader.resolution 2 --scene.patch_size 2 ---scene.gaussians.densify-abs-grad-threshold 0.00015 --scene.gaussians.opacity-cull-threshold 0.05"
    
    for scene in tnt_360_scenes:
        if args.method.endswith('2dgs'):
            common_args_ = f"{common_args} --scene.lambda_dist 100"
        else:
            common_args_ = common_args
        source = os.path.join(args.TNT_data, scene)
        print(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args_}")
        os.system(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args_}")
    
    for scene in tnt_large_scenes:
        if args.method.endswith('2dgs'):
            common_args_ = f"{common_args} --scene.lambda_dist 10"
        else:
            common_args_ = common_args
        source = os.path.join(args.TNT_data, scene)
        print(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args_}")
        os.system(f"python train.py {args.method} --source-path {source} --output_path {args.output_path} --experiment_name {scene} --timestamp './' {common_args_}")


if not args.skip_rendering:
    for scene in tnt_360_scenes:
        common_args = "--num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
        config = os.path.join(args.output_path, scene, args.method, 'config.yml')
        source = os.path.join(args.TNT_data, scene)
        print(f"python extract_mesh.py --load_config {config} {common_args}")
        os.system(f"python extract_mesh.py --load_config {config} {common_args}")

    for scene in tnt_large_scenes:
        common_args = "--num_cluster 1 --voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5"
        config = os.path.join(args.output_path, scene, args.method, 'config.yml')
        source = os.path.join(args.TNT_data, scene)
        print(f"python extract_mesh.py --load_config {config} {common_args}")
        os.system(f"python extract_mesh.py --load_config {config} {common_args}")


if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_scenes = tnt_360_scenes + tnt_large_scenes

    for scene in all_scenes:
        ply_file = os.path.join(args.output_path, scene, args.method, 'train/ours_30000/fuse_post.ply')
        string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
            f"--dataset-dir {args.TNT_GT}/{scene} " + \
            f"--traj-path {args.TNT_data}/{scene}/{scene}_COLMAP_SfM.log " + \
            f"--ply-path {ply_file} " + \
            f"--out-dir {script_dir}/tmp-tnt/{args.method}/{scene}"
        print(string)
        os.system(string)