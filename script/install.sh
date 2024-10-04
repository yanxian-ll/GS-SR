## Please install cuda-toolkit first

## crate conda-environment
conda create --name 3dgs python=3.8
conda activate 3dgs

## install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install plyfile tqdm scipy shapely einops lpips torch_scatter jaxtyping opencv-python open3d tensorboard mediapy

## install submodules
pip install submodules/simple-knn
pip install submodules/diff-gaussian-rasterization
pip install submodules/diff-plane-rasterization
pip install submodules/diff-surfel-rasterization
pip install submodules/scaffold-filter
