#!/bin/bash
PYTHON='/home/gui/.conda/envs/mmcv2/bin/python'
export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/home/gui/Downloads/gyy0525/log/tiny-imagenet/
LOSS='Softmax' # For TinyImageNet, ARPLoss and Softmax loss have the same
                     # RandAug and Label Smoothing hyper-parameters, but different learning rates

# Fixed hyper params for both ARPLoss and Softmax
LABEL_SMOOTHING=0

# LR different for ARPLoss and others
if [ $LOSS = "ARPLoss" ]; then
   LR=0.001
else
   LR=0.01
fi

# tinyimagenet
for SPLIT_IDX in 0; do

  EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
  EXP_NUM=$((${EXP_NUM}+1))
  echo $EXP_NUM

  ${PYTHON} -m osr   --lr=${LR} \
                     --model='resnet18' \
                     --transform='tinyimagenet' \
                     --dataset='tinyimagenet' \
                     --image_size=64 \
                     --loss=${LOSS} \
                     --scheduler='cosine_warm_restarts_warmup' \
                     --label_smoothing=${LABEL_SMOOTHING} \
                     --split_train_val='False' \
                     --batch_size=128 \
                     --num_workers=12 \
                     --max-epoch=100 \
                     --seed=0 \
                     --gpus 0 \
                     --weight_decay=1e-4 \
                     --num_restarts=2 \
                     --feat_dim=512 \
                     --split_idx=${SPLIT_IDX} \
                     > ${SAVE_DIR}logfile_${EXP_NUM}.out

done