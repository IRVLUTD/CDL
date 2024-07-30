#!/bin/bash
# Experiment settings

# Build our CDL model on CODAPrompt baseline(DualPrompt and L2P)
# To ensure a fair comparison, we will make the parameters for CODAPrompt, DualPrompt, and L2P consistent with those in the experimental code of CODA-Prompt:
# https://github.com/GT-RIPL/CODA-Prompt

DATASET=ImageNet_R
N_CLASS=200

# Hard coded inputs
GPUID='0 1'
CONFIG=configs/imnet-r_prompt.yaml
OVERWRITE=0
RANDOM_SEED=1

# Adjust Model
T_MODEL='vit_base_patch16_224'
S_MODEL='vit_small_patch16_224'

# Get the current time
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# Save directory
OUTDIR=${CURRENT_TIME}_$T_MODEL_$S_MODEL_${RANDOM_SEED}/${DATASET}/10-task

###############################################################

mkdir -p $OUTDIR

# CODA-P
# prompt parameter args:
#    arg 1 = prompt component pool size
#    arg 2 = prompt length
#    arg 3 = ortho penalty loss weight
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED

# DualPrompt
# prompt parameter args:
#    arg 1 = e-prompt pool size (# tasks)
#    arg 2 = e-prompt pool length
#    arg 3 = g-prompt pool length
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name DualPrompt \
    --prompt_param 10 20 6 \
    --log_dir ${OUTDIR}/dual-prompt \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED

# L2P
# # prompt parameter args:
# #    arg 1 = e-prompt pool size (# tasks)
# #    arg 2 = e-prompt pool length
# #    arg 3 = -1 -> shallow
python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
    --learner_type prompt --learner_name L2P \
    --prompt_param 30 20 -1 \
    --log_dir ${OUTDIR}/l2p \
    --t_model $T_MODEL \
    --s_model $S_MODEL \
    --random_s $RANDOM_SEED

