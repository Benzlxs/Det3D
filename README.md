# Det3D

A general 3D Object Detection codebase in PyTorch

## Call for contribution.
* Support Waymo Dataset.
* Add other 3D detection / segmentation models, such as VoteNet, STD, etc.

## Introduction

Det3D is the first 3D Object Detection toolbox which provides off the box implementations of many 3D object detection algorithms such as PointPillars, SECOND, PIXOR, etc, as well as state-of-the-art methods on major benchmarks like KITTI(ViP) and nuScenes(CBGS). Key features of Det3D include the following aspects:

* Multi Datasets Support: KITTI, nuScenes, Lyft
* Point-based and Voxel-based model zoo
* State-of-the-art performance
* DDP & SyncBN


## Installation

Please refer to [INSTALATION.md](INSTALLATION.md).

## Quick Start

Please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## Model Zoo and Baselines
### [3DBN](https://github.com/Benzlxs/Det3D/blob/master/examples/tdbn/config/tdbn2_det2_vef_70.py) on KITTI(val) Dataset 3:1
```
bbox AP:90.55, 89.42, 88.24
bev  AP:90.20, 88.30, 79.59
3d   AP:89.43, 85.48, 77.36
aos  AP:89.85, 88.14, 86.94
```


### To Be Released

1. [PointPillars](examples/point_pillars/configs/nusc_all_point_pillars_mghead_syncbn.py) on NuScenes(val) Dataset
2. [CGBS](examples/cbgs/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on NuScenes(val) Dataset
3. [CGBS](examples/cbgs/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py) on Lyft(val) Dataset

## Currently Support

* Models
  - [x] VoxelNet
  - [x] SECOND
  - [x] PointPillars
* Features
    - [x] Multi task learning & Multi-task Learning
    - [x] Distributed Training and Validation
    - [x] SyncBN
    - [x] Flexible anchor dimensions
    - [x] TensorboardX
    - [x] Checkpointer & Breakpoint continue
    - [x] Self-contained visualization
    - [x] Finetune
    - [x] Multiscale Training & Validation
    - [x] Rotated RoI Align


## TODO List
* Models
  - [ ] PointRCNN
  - [ ] PIXOR

## Developers

[Benjin Zhu](https://github.com/poodarchu/) , [Bingqi Ma](https://github.com/a157801)

## License

Det3D is released under the [Apache licenes](LICENES).

## Acknowledgement

* [mmdetection](https://github.com/open-mmlab/mmdetection) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
