#!/bin/bash
# Experiment settings

# Build our CDL model on CODAPrompt baseline(DualPrompt, L2P and APT)


DATASET=cifar-100
N_CLASS=100

# Hard coded inputs
GPUID='0 1'
CONFIG=configs/cifar-100_prompt.yaml
OVERWRITE=0
RANDOM_SEED=1


# Adjust Model
T_MODEL='vit_base_patch16_224'
S_MODEL='vit_small_patch16_224'

# T_MODEL='vit_large_patch16_224'
# S_MODEL='vit_base_patch16_224'

# Get the current time
CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

# Get the KD methods
KD_METHOD='KD_Token'     # [KD_Token', 'KD', 'DKD', 'FitNets', 'ReviewKD']

# DCDL parameters # Get the KD_Prompt parameters (kd_layers size, kd_prompt_length) 
KD_Prompt_Param='12 6'
KD_ALPHA=0.5

# APT parameters
APT_PROMPT_DROPOUT='0.01'
EMA_COEFF='0.7'


# # Save directory
OUTDIR=Results/${CURRENT_TIME}_${T_MODEL}_${S_MODEL}_${RANDOM_SEED}_${DATASET}_${KD_METHOD}_${KD_ALPHA}/${DATASET}/10-task

# ###############################################################

# mkdir -p $OUTDIR




# APT
# prompt parameter args:
#    arg 1 = prompt dropout ratio
# python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name APT \
#     --prompt_param $APT_PROMPT_DROPOUT \
#     --ema_coeff $EMA_COEFF \
#     --log_dir ${OUTDIR}/apt \
#     --t_model $T_MODEL \
#     --s_model $S_MODEL \
#     --random_s $RANDOM_SEED \
#     --KD_method $KD_METHOD \
#     --kd_prompt_param $KD_Prompt_Param \
#     --kd_alpha $KD_ALPHA



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
    --random_s $RANDOM_SEED \
    --KD_method $KD_METHOD \
    --kd_prompt_param $KD_Prompt_Param


# # DualPrompt
# # prompt parameter args:
# #    arg 1 = e-prompt pool size (# tasks)
# #    arg 2 = e-prompt pool length
# #    arg 3 = g-prompt pool length
# python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt \
#     --t_model $T_MODEL \
#     --s_model $S_MODEL \
#     --random_s $RANDOM_SEED \
#     --KD_method $KD_METHOD \
#     --kd_prompt_param $KD_Prompt_Param

# # L2P
# # # prompt parameter args:
# # #    arg 1 = e-prompt pool size (# tasks)
# # #    arg 2 = e-prompt pool length
# # #    arg 3 = -1 -> shallow
# python -u run.py --config $CONFIG --gpuid $GPUID --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p \
#     --t_model $T_MODEL \
#     --s_model $S_MODEL \
#     --random_s $RANDOM_SEED \
#     --KD_method $KD_METHOD \
#     --kd_prompt_param $KD_Prompt_Param



