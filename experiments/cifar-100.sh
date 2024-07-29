# bash experiments/cifar-100.sh
# experiment settings
DATASET=cifar-100
N_CLASS=200

# save directory
OUTDIR=outputs_test_0728_t_prompt_s_prompt_projectCODAPrompt_No_Kd_tokens_buthave_kdlast_useKLDivLoss_largetobase_cifar100_03/${DATASET}/10-task

# hard coded inputs
GPUID='0 1'
CONFIG=configs/cifar-100_prompt.yaml
CONFIG_FT=configs/cifar-100_ft.yaml
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
    --t_model 'vit_large_patch16_224' \
    --s_model 'vit_base_patch16_224' \
    --random_s 7

# DualPrompt

# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name DualPrompt \
#     --prompt_param 10 20 6 \
#     --log_dir ${OUTDIR}/dual-prompt \
#     --t_model 'vit_base_patch16_224' \
#     --s_model 'vit_tiny_patch16_224'

# # # L2P

# python -u run.py --config $CONFIG --gpuid $GPUID --repeat $REPEAT --overwrite $OVERWRITE \
#     --learner_type prompt --learner_name L2P \
#     --prompt_param 30 20 -1 \
#     --log_dir ${OUTDIR}/l2p \
#     --t_model 'vit_base_patch16_224' \
#     --s_model 'vit_tiny_patch16_224'



