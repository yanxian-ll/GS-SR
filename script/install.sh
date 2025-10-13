## Please install cuda-toolkit first

## crate conda-environment
conda create --name 3dgs python=3.8
conda activate 3dgs

## install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install plyfile tensorboard tqdm einops wandb lpips laspy \
    jaxtyping colorama opencv-python scikit-learn trimesh open3d pytorch3d \
    matplotlib mediapy opencv_python Pillow PyYAML rich scipy Shapely \
    torch_scatter torchtyping jaxtyping tyro scikit-image rasterio numba plyflatten pyproj

## install submodules
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-plane-rasterization
pip install submodules/diff-surfel-rasterization
pip install submodules/scaffold-filter
