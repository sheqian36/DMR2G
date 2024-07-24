#!/bin/bash
set -u


# GPU=${1}
GPU=1
NUM_GPUS=1
LOSS_FUNC="uniform"
# SRC=${2:-'before'}
# TGT=${3:-'after'}
LR=0.0001
SEQ_LEN=64
WARMUP=10000
EVAL_INTERVAL=10000
SCHEDULE_UPDATE_STRIDE=10000
DSET="iu_xray_ckpts"
UPDATE_GRANU=20
INIT_PRETRAINED_MODEL="False"
USE_PRETRAINED_EMBEDDINGS="False"
FREEZE_EMBEDDINGS="False"
LR_ANNEAL_STEPS=80000
DIFFUSION_STEPS=2000
NOISE_SCHEDULE=sqrt
BATCH_SIZE=64


# CHECKPOINT_PATH="ckpts/${DSET}/debug"
CHECKPOINT_PATH="/home/shuchenweng/cz/oyh/output/seqdiffuseq/${DSET}/debug"
TRAIN_TXT_PATH="/home/shuchenweng/cz/oyh/data/seqdiffuseq/iu_xray/train"
VAL_TXT_PATH="/home/shuchenweng/cz/oyh/data/seqdiffuseq/iu_xray/val"
TEST_TXT_PATH="/home/shuchenweng/cz/oyh/data/seqdiffuseq/iu_xray/test"
LABEL_PATH="/home/shuchenweng/cz/oyh/data/seqdiffuseq/iu_xray/labels_14.pickle"
# INIT_PROTYPES_PATH="/home/oyh2024/project/data/seqdiffuseq/iu_xray/init_prototypes.pt"
TASK="iu_xray"
IN_CHANNELS=512
WEIGHT_DECAY=0.0
SEED=2023
DROPOUT=0.3
NUM_HEADS=8
# CONFIG_NAME="/home/shuchenweng/cz/oyh/model/seqdiffuseq/gpt2"   #"facebook/bart-base"
CONFIG_NAME="/home/shuchenweng/cz/oyh/model/seqdiffuseq/bart"   #"facebook/bart-base"
NOTES="iu_xray training with noise schedule and self condition"

mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${CHECKPOINT_PATH}/log/
export DIFFUSION_BLOB_LOGDIR=${CHECKPOINT_PATH}/log/


ARGS=(--checkpoint_path ${CHECKPOINT_PATH}
    --save_interval ${WARMUP}
    --eval_interval ${EVAL_INTERVAL} 
    --lr ${LR}
    --batch_size ${BATCH_SIZE}
    # --src ${SRC}
    # --tgt ${TGT}
    --diffusion_steps ${DIFFUSION_STEPS}
    --noise_schedule ${NOISE_SCHEDULE}
    --sequence_len ${SEQ_LEN} --seed ${SEED}
    --weight_decay ${WEIGHT_DECAY}
    --predict_xstart True
    --train_txt_path ${TRAIN_TXT_PATH}
    --dataset "iu_xray"
    --val_txt_path ${VAL_TXT_PATH}
    --test_txt_path ${TEST_TXT_PATH}
    --label_path ${LABEL_PATH}
    # --init_protypes_path ${INIT_PROTYPES_PATH}
    --task ${TASK}
    --config_name ${CONFIG_NAME}
    --init_pretrained ${INIT_PRETRAINED_MODEL}
    --freeze_embeddings ${FREEZE_EMBEDDINGS}
    --use_pretrained_embeddings ${USE_PRETRAINED_EMBEDDINGS}
    --notes \""${NOTES}"\")

if [ ${LR_ANNEAL_STEPS} -eq 0 ]; then
    LR_ANNEAL_STEPS=100
    DEBUG=true
else
    DEBUG=false
fi

ARGS+=(--lr_anneal_steps $LR_ANNEAL_STEPS)

if [ $DEBUG = true ]; then
    ARGS+=(--debug)
fi

ARGS+=(--encoder_layers 6
    --decoder_layers 12
    --num_heads 8
    --num_heads 8
    --in_channel 512
    --out_channel 512
    --num_channels 2048
    --sequence_len_src 128
    --warmup $WARMUP
    --schedule_sampler $LOSS_FUNC
    --loss_update_granu $UPDATE_GRANU
    --schedule_update_stride $SCHEDULE_UPDATE_STRIDE)

export CUDA_VISIBLE_DEVICES=$GPU && mpiexec --allow-run-as-root -n $NUM_GPUS python -u main.py "${ARGS[@]}"
# export CUDA_VISIBLE_DEVICES=$GPU python -u main.py "${ARGS[@]}"
