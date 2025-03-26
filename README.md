##  Continual Distillation Learning
PyTorch code for the paper:\
**Continual Distillation Learning: Knowledge Distillation in Prompt-based Continual Learning**\
Qifan Zhang, Yunhui Guo, Yu Xiang

[arXiv](https://arxiv.org/abs/2407.13911), [Project](https://irvlutd.github.io/CDL/)

<p align="center">
<img src="CDL_framework.png" width="90%">
</p>

## Abstract
We introduce the problem of continual distillation learning (CDL) in order to use knowledge distillation (KD) to improve prompt-based continual learning (CL) models. The CDL problem is valuable to study since the use of a larger vision transformer (ViT) leads to better performance in prompt-based continual learning. The distillation of knowledge from a large ViT to a small ViT can improve the inference efficiency for prompt-based CL models. We empirically found that existing KD methods such as logit distillation and feature distillation cannot effectively improve the student model in the CDL setup. To this end, we introduce a novel method named Knowledge Distillation based on Prompts (KDP), in which globally accessible prompts specifically designed for knowledge distillation are inserted into the frozen ViT backbone of the student model. We demonstrate that our KDP method effectively enhances the distillation performance in comparison to existing KD methods in the CDL setup.


## Setup
 * set up conda environment w/ python 3.8, ex: `conda create --name CDL python=3.8`
 * `conda activate CDL`
 * `sh install_requirements.sh`
 
## Datasets
 * Create a folder `data/`
 * **CIFAR 100**: should automatically be downloaded
 * **ImageNet-R**: retrieve from: https://github.com/hendrycks/imagenet-r
 * **./data** should be a folder with the following structure:
```bash
  ./data  
  ├── cifar-100-python  
  ├── imagenet-r  
  │   ├── n01443537  
  │   │   ├── art_0.jpg  
  │   │   ├── cartoon_0.jpg  
  │   │   ├── graffiti_0.jpg
  │   │   └── ...
  │   ├── n01833805  
  │   │   ├── art_0.jpg  
  │   │   ├── cartoon_0.jpg  
  │   │   ├── graffiti_0.jpg
          └── ... 
```

## Training
**The scripts are set up for 2 GPUs** but can be modified for your hardware. You can directly run the run.py and test on ImageNet-R dataset:

```bash
python -u run.py --config configs/imnet-r_prompt.yaml --gpuid 0 1 \
    --learner_type prompt --learner_name CODAPrompt \
    --prompt_param 100 8 0.0 \
    --log_dir ImageNet_R/coda-p \
    --t_model 'vit_base_patch16_224' \
    --s_model 'vit_small_patch16_224' \
    --KD_method 'KD_Token' \
    --kd_prompt_param 12 6
```
* Check the experiments/imagenet-r.sh and experiments/cifar-100.sh to see the details.
* You can change the learner_name for DualPrompt or L2P.
* Change the prompt_param for different learner(CODA, DualPrompt or L2P)
* You can adjust the teacher and student's model with --t_model and --s_model.
* Change the --KD_method for different knowledage distillation methods -> ['KD_Token', 'KD', 'DKD', 'FitNets', 'ReviewKD']. Use the 'KD_Token' for our ***KDP*** model.
* Change the --kd_prompt_param for our ***KDP*** model (kd_layers size, kd_prompt_length).


## Results

The results will be saved in the created --log_dir folder, including the models for the teacher and student as well as the final average accuracy for both the teacher and student.

<!-- ## Citation
If you find the method useful in your research, please consider citing:
```latex
@misc{lu2024adapting,
    title={Continual Distillation Learning},
    author={Qifan Zhang and Yunhui Guo and Yu Xiang},
    year={2024},
    eprint={2407.13911},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
``` -->

## Acknowledgments

This project is based on the following repositories:
- [CODA-Prompt](https://github.com/GT-RIPL/CODA-Prompt)
- [L2P-Pytorch](https://github.com/JH-LEE-KR/l2p-pytorch)



