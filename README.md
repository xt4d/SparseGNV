# SparseGNV: Generating Novel Views of Indoor Scenes with Sparse Input Views

### SparseGNV generates novel view images of indoor scenes based on 4 observed views:
<img src="images/show.png" width="50%"/>

### [[ArXiv Paper](https://arxiv.org/abs/2305.07024)] </h2>
NOTE: This code repository provides a streamlined version of SparseGNV. Specifically, the neural geometry module originally designed with [Point-NeRF](https://github.com/Xharlie/pointnerf) is modified to using raw point clouds with point-based renderer, as the code based on Point-NeRF is overly intricate for integration.

## TL;DR
- SparseGNV generates novel views of indoor scenes given sparse RGB-D input views.
- SparseGNV consists of three modules: 1) a neural point cloud (Point-NeRF) built from input views to project visual guidance; 2) an autoregressive transformer to generate image tokens of the target view conditioned on visual guidance; 3) a VQ decoder to recover the RGB image given the generated image tokens.

## Setup Environment
Step 1: Create a python environment using Anaconda or Miniconda:
```
conda create -n sparsegnv python=3.10
conda activate sparsegnv
```
Step 2: Install PyTorch3D (including PyTorch) following [Official Instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). 

Step 3: Install additional packages:
```
pip install -r requirements.txt
```

## Download Checkpoints
Download the [VQ decoder model](https://drive.google.com/file/d/1fD2fPq5hjwLcC3yJ019SPqle8gOBfG_N/view?usp=sharing) and the [generator model](https://drive.google.com/file/d/1RgT6zikjrM2vfppCIKPsGu3kxcuomfOx/view?usp=sharing). Put the checkpoints under ``ckpts/``.

## Run Demo
The demo is tested on an NVIDIA Tesla V100 32G GPU.

Step 1: Render hints with sparse RGB-D inputs and novel camera poses:
```
python render_hint.py --data_root data/scene0710_00
```

Step 2: Generate novel view images:
```
python generate.py --data_root data/scene0710_00
```

## Prepare your own data
The organization of the testing data follows [ScanNet](http://www.scan-net.org/). We provide an example in ``data/scene0710_00/``. 
- The captured data are placed under ``exported/`` with color images, depth images, camera intrinsics, and camera poses (cam2world matrix in the Blender format). 
- The poses of targeting novel views are placed under ``novel_pose/``. 
- ``obs_vids.txt`` provides the frame names of the observed images and poses. 
- ``novel_vids.txt`` provides the frame names of the novel poses. 

Please make sure that all the frame names are numbers and unique.



##  Citation
```
@article{cheng2023sparsegnv,
  title={SparseGNV: Generating Novel Views of Indoor Scenes with Sparse Input Views},
  author={Cheng, Weihao and Cao, Yan-Pei and Shan, Ying},
  journal={arXiv preprint arXiv:2305.07024},
  year={2023}
}
``` 
