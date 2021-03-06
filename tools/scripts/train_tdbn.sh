#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
OUT_DIR=/home/ben/projects/Det3D/training_results

TDBN_WORK_DIR=$OUT_DIR/TDBN_$TASK_DESC\_$DATE_WITH_TIME


# Voxelnet

python ./tools/train.py examples/tdbn/config/kitti_car_vfev3_tdbn1_det2.py --work_dir=$TDBN_WORK_DIR
#pudb3 ./tools/train.py examples/tdbn/config/kitti_car_vfev3_tdbn1_det2.py --work_dir=$TDBN_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=4 ./tools/train.py examples/tdbn/config/kitti_car_vfev3_tdbn1_det2.py --work_dir=$TDBN_WORK_DIR
