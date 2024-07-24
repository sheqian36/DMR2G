#!/bin/bash


# MODEL_NAME=$1
MODEL_NAME="/home/shuchenweng/cz/oyh/output/seqdiffuseq/mimic_cxr_ckpts/debug/ema_0.9999_180000.pt"
OUT_DIR=$2
# SCHEDULE_PATH=$3
SCHEDULE_PATH="/home/shuchenweng/cz/oyh/output/seqdiffuseq/mimic_cxr_ckpts/debug/alpha_cumprod_step_180000.npy"
if [ -z "$OUT_DIR" ]; then
    OUT_DIR=${MODEL_NAME}
fi

SEED=${4:-10708}
GEN_BY_Q=${5:-"False"}
GEN_BY_MIX=${6:-"False"}
MIX_PROB=${7:-0}
MIX_PART=${8:-1}
TOP_P=-1
CLAMP="no_clamp"
BATCH_SIZE=64
DIFFUSION_STEPS=2000
NUM_SAMPLES=-1
SEQ_LEN=100
VAL_TXT="/home/shuchenweng/cz/oyh/data/seqdiffuseq/mimic_cxr/test"
TASK="mimic_cxr"
DATASET="mimic_cxr"

CUDA_VISIBLE_DEVICES=1 python -u inference_main.py --model_name_or_path ${MODEL_NAME} --sequence_len_src 128 \
--batch_size ${BATCH_SIZE} --num_samples ${NUM_SAMPLES} --top_p ${TOP_P} --time_schedule_path ${SCHEDULE_PATH} \
--seed ${SEED} --test_txt_path ${VAL_TXT} --task ${TASK} --dataset ${DATASET} --generate_by_q ${GEN_BY_Q} --generate_by_mix ${GEN_BY_MIX} \
--out_dir ${OUT_DIR} --diffusion_steps ${DIFFUSION_STEPS} --clamp ${CLAMP} --sequence_len ${SEQ_LEN} --generate_by_mix_prob ${MIX_PROB} --generate_by_mix_part ${MIX_PART}
