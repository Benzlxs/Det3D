#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
#CONFIG=examples/second/configs/tdbn1_det2_vef_60.py
CONFIG=examples/tdbn/config/tdbn2_det2_bv_50.py
#CHECKPOINT="/home/ben/projects/Det3D/training_results/vef_tdbn1_60_110_redo/epoch_110.pth"
CHECKPOINT="/home/ben/projects/Det3D/training_results/SECOND__20200429-123016/epoch_76.pth"
WORK_DIR="/home/ben/projects/Det3D/training_results/SECOND__20200429-123016"
#WORK_DIR="/home/ben/projects/Det3D/training_results/vef_tdbn1_60_110_redo"

# Test

#exe=python
exe=pudb3

script="$exe ./tools/dist_test.py $CONFIG --checkpoint=$CHECKPOINT --work_dir=$WORK_DIR"
$script




#pudb3 ./tools/dist_test.py \
#    $CONFIG \
#    --work_dir=$WORK_DIR \
#    --checkpoint=$CHECKPOINT \


