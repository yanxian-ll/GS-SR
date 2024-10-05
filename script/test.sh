
source-path=""

## test vanilla-version
# 3dgs
python train.py -s ${source-path}
# scaffold-gs
python train.py -s ${source-path}
# octree-gs
python train.py -s ${source-path}
# 2dgs
python train.py -s ${source-path} --depth_ratio 0 --opacity_cull 0.05
# pgsr
python train.py -s ${source-path} --max_abs_split_points 0 --opacity_cull_threshold 0.05


## test GS-SR version
# 3dgs
python train.py 3dgs --source-path ${source-path}
# scaffold-gs
python train.py scaffold-gs --source-path ${source-path}
# octree-gs
python train.py octree-gs --source-path ${source-path}
# 2dgs
python train.py 2dgs --source-path ${source-path} --scene.depth-ratio 0 --scene.gaussians.opacity-cull-threshold 0.05
# pgsr
python train.py pgsr --source-path ${source-path} --scene.gaussians.max-abs-split-points 0 --scene.gaussians.opacity_cull_threshold 0.05

