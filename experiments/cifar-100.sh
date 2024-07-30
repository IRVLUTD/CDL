#!/bin/bash
# Experiment settings
DATASET=cifar-100
N_CLASS=200

# hard coded inputs
GPUID='0 1'
CONFIG=configs/cifar-100_prompt.yaml
OVERWRITE=0
RANDOM_SEED=1

# Adjust Model
T_MODEL='vit_base_patch16_224'
S_MODEL='vit_tiny_patch16_224'

# Get the current time
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# Save directory
OUTDIR=${CURRENT_TIME}_$T_MODEL_$S_MODEL_${RANDOM_SEED}/${DATASET}/10-task

###############################################################

mkdir -p $OUTDIR

# CODA-P
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED

# DualPrompt
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 \
    --log_dir ${OUTDIR}/dual-prompt \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED

# L2P
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir ${OUTDIR}/l2p \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED



