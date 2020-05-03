#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/ben/projects/Det3D/training_results

SECOND_WORK_DIR=$OUT_DIR/SECOND_$TASK_DESC\_$DATE_WITH_TIME

config=examples/tdbn/config/tdbn1_det2_vef_50.py

# Voxelnet

exe=python
#exe=pudb3

script="$exe ./tools/train.py $config --work_dir=$SECOND_WORK_DIR"
$script


#python ./tools/train.py examples/tdbn/config/tdbn1_det2_vef_50.py --work_dir=$SECOND_WORK_DIR
#pudb3 ./tools/train.py examples/tdbn/config/tdbn1_det2_bv.py --work_dir=$SECOND_WORK_DIR

# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/tdbn/config/kitti_car_vfev3_tdbn1_det2.py --work_dir=$SECOND_WORK_DIR
