# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
#OUTDIR=outputs/${DATASET}/10-task

OUTDIR=outputs_test_0726_t_prompt_s_prompt_projectCODAlayer_No_Kd_tokens_buthave_kdlast_useKLDivLoss_basetosmall_epoch35_test_03_modify01/${DATASET}/10-task

# hard coded inputs
GPUID='0 1'
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=1
OVERWRITE=0



###############################################################

# process inputs
mkdir -p $OUTDIR




# CODA-P

python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ${OUTDIR}/coda-p \
    --t_model 'vit_base_patch16_224' \
    --s_model 'vit_small_patch16_224'

# DualPrompt

# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt \
#     --t_model 'vit_base_patch16_224' \
#     --s_model 'vit_small_patch16_224'

# # # L2P

# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p \
#     --t_model 'vit_base_patch16_224' \
#     --s_model 'vit_small_patch16_224'