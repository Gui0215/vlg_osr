#!/bin/bash
PYTHON='/home/gui/.conda/envs/mmcv2/bin/python'
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/home/gui/Downloads/vlg_osr/log/tiny-imagenet/
LOSS='SupConLoss'
LR=0.01
TEMP=0.1
# Fixed hyper params for both ARPLoss and Softmax
LABEL_SMOOTHING=0

# tinyimagenet
for SPLIT_IDX in 0; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m pretrain_osr   --lr=${LR} \
                     --model='vgg32' \
                     --temp=${TEMP} \
                     --transform='rand-augment' \
                     --rand_aug_m=8 \
                     --rand_aug_n=1 \
                     --dataset='tinyimagenet' \
                     --image_size=64 \
                     --loss=${LOSS} \
                     --scheduler='cosine' \
                     --split_train_val='False' \
                     --batch_size=256 \
                     --num_workers=12 \
                     --max-epoch=300 \
                     --seed=0 \
                     --gpus 0 \
                     --weight_decay=1e-4 \
                     --feat_dim=128 \
                     --split_idx=${SPLIT_IDX} \
                     > ${SAVE_DIR}logfile_${EXP_NUM}.out

done