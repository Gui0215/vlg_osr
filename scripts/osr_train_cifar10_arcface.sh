#!/bin/bash
PYTHON='/home/gui/.conda/envs/mmcv2/bin/python'
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/home/gui/Downloads/gyy0525/log/cifar10/

LOSS='ArcFace'
LR=0.01


for SPLIT_IDX in 0 1 2 3 4; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m osr --lr=${LR} \
                   --model='vgg32' \
                   --dataset='cifar-10-10' \
                   --transform='cifar' \
                   --image_size=32 \
                   --optim='sgd' \
                   --loss=${LOSS} \
                   --batch_size=128 \
                   --scheduler='multi_step' \
                   --num_workers=12 \
                   --gpus 0 \
                   --max-epoch=100 \
                   --seed=0 \
                   --gpus 0 \
                   --feat_dim=128 \
                   --split_idx=${SPLIT_IDX} \
                   > ${SAVE_DIR}logfile_${EXP_NUM}.out

done