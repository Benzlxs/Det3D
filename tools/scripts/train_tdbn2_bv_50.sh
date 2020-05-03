#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/ben/projects/Det3D/training_results

TDBN_WORK_DIR=$OUT_DIR/TDBN_$TASK_DESC\_$DATE_WITH_TIME

config=examples/tdbn/config/tdbn2_det2_bv_50.py

# Voxelnet

exe=python
#exe=pudb3

script="$exe ./tools/train.py $config --work_dir=$TDBN_WORK_DIR"
$script


#python ./tools/train.py examples/tdbn/config/tdbn1_det2_vef_50.py --work_dir=$TDBN_WORK_DIR
#pudb3 ./tools/train.py examples/tdbn/config/tdbn1_det2_bv.py --work_dir=$TDBN_WORK_DIR

# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/tdbn/config/kitti_car_vfev3_tdbn1_det2.py --work_dir=$TDBN_WORK_DIR
