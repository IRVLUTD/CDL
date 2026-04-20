'''
 * Based on coda prompt here
 * https://github.com/GT-RIPL/CODA-Prompt
 * Build our CDL model on CODAPrompt baseline(DualPrompt and L2P)
'''

from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']
        super(Prompt, self).__init__(learner_config)

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask


    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask
    
    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def dkd_loss(self, logits_student, logits_teacher, target, alpha, beta, temperature):

        gt_mask = self._get_gt_mask(logits_student, target)
        other_mask = self._get_other_mask(logits_student, target)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher.detach(), size_average=False)
            * (temperature**2)
            / target.shape[0]
        )


        
        pred_teacher_part2 = F.softmax(
            logits_teacher / temperature - 1000.0 * gt_mask, dim=1
        )
        log_pred_student_part2 = F.log_softmax(
            logits_student / temperature - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2.detach(), size_average=False)
            * (temperature**2)
            / target.shape[0]
        )

        return alpha * tckd_loss + beta * nckd_loss




    def distillation_loss(self, y_student, y_teacher, t=2, alpha=0.5):
        soft_loss = F.kl_div(F.log_softmax(y_student / t, dim=1),
                            F.softmax(y_teacher / t, dim=1).detach(),
                            reduction='batchmean') * (t * t * alpha)
        
        
        return soft_loss
    def update_model(self, inputs, targets, epoch):


        # logits
        logits, prompt_loss, p_list_, t_corr_list, t_features_list = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]

        cur_logits = logits[:,self.last_valid_out_dim:self.valid_out_dim]
        #cur_logits = logits[:,:self.valid_out_dim].clone()

        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        # ce loss
        total_loss = total_loss + prompt_loss.sum()

        # step
        if(epoch<=50):
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

   
        return total_loss.detach(), logits, cur_logits, p_list_, t_corr_list, t_features_list
    
    # predicts_ is the current task's logits of teacher model
    def s_update_model(self, inputs, targets, predicts_, p_list_, t_corr_list_, t_features_list):

 

        p_list_ = None
        logits, kd_logits, prompt_loss, prompt_loss_ , feature_loss= self.s_model(inputs, train=True, t_p_list_ = p_list_, t_corr_list_ = t_corr_list_, t_features_list_ = t_features_list)

        if self.config['KD_method'] == 'KD_Token':
            total_logits = (1-self.config['kd_alpha'])*logits+self.config['kd_alpha']*kd_logits
            logits = logits[:,:self.valid_out_dim]
            cur_logits = kd_logits[:,self.last_valid_out_dim:self.valid_out_dim]
        else:
            total_logits = logits
            logits = logits[:,:self.valid_out_dim]

            cur_logits = logits[:,self.last_valid_out_dim:self.valid_out_dim].clone()


  
    
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        
        if self.config['KD_method'] == 'DKD':
            cur_targets = targets - self.last_valid_out_dim
            soft_loss = self.config['kd_alpha']*self.dkd_loss(cur_logits, predicts_, cur_targets, 1, 4, 4)
        else:
            soft_loss = self.distillation_loss(cur_logits, predicts_, t=self.config['Soft_T'], alpha=self.config['kd_alpha'])

       
 
        total_loss = (1-self.config['kd_alpha'])*self.criterion(logits, targets.long(), dw_cls) + soft_loss

        
        total_loss = total_loss + prompt_loss.sum() +  feature_loss.sum()

        # step
        self.s_optimizer.zero_grad()
        total_loss.backward()
        self.s_optimizer.step()





        soft_loss_cur=0
        soft_loss_pre=0
        
      
        return total_loss.detach(), soft_loss, soft_loss_cur, soft_loss_pre, prompt_loss_.sum(), feature_loss.sum(), total_logits

    # sets model optimizers
    def init_optimizer(self):


        # parse optimizer args
        # Multi-GPU
        # if len(self.config['gpuid']) > 1:
        #     params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
        # else:
        #     params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())

        if len(self.config['gpuid']) > 1:
            if self.model.module.prompt_flag == 'apt':
                params_to_opt = (
                    list(self.model.module.prompt.parameters()) +
                    list(self.model.module.last.parameters()) 
                    # list(self.model.module.clf_norm.parameters())
                )
            else:
                params_to_opt = (
                    list(self.model.module.prompt.parameters()) +
                    list(self.model.module.last.parameters())
                )
        else:
            if self.model.prompt_flag == 'apt':
                params_to_opt = (
                    list(self.model.prompt.parameters()) +
                    list(self.model.last.parameters())
                    # list(self.model.clf_norm.parameters())
                )
            else:
                params_to_opt = (
                    list(self.model.prompt.parameters()) +
                    list(self.model.last.parameters())
                )


        print('*****************************************')
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)



        # if len(self.config['gpuid']) > 1:
        #     if self.config['KD_method'] == 'KD_Token' :
        #         s_params_to_opt = list(self.s_model.module.prompt.parameters()) + list(self.s_model.module.last.parameters())  + list(self.s_model.module.kd_last.parameters()) + list(self.s_model.module.kd_prompt.parameters())
        #     elif self.config['KD_method'] == 'KD' :
        #         s_params_to_opt = list(self.s_model.module.prompt.parameters()) + list(self.s_model.module.last.parameters()) 
        #     elif self.config['KD_method'] == 'DKD' :
        #         s_params_to_opt = list(self.s_model.module.prompt.parameters()) + list(self.s_model.module.last.parameters())
        #     elif self.config['KD_method'] == 'FitNets' :
        #         s_params_to_opt = list(self.s_model.module.prompt.parameters()) + list(self.s_model.module.last.parameters()) + list(self.s_model.module.project_t_s.parameters())
        #     elif self.config['KD_method'] == 'ReviewKD' :
        #         s_params_to_opt = list(self.s_model.module.prompt.parameters()) + list(self.s_model.module.last.parameters()) + list(self.s_model.module.ReviewKD_layers.parameters())
        # else:
        #     if self.config['KD_method'] == 'KD_Token' :
        #         s_params_to_opt = list(self.s_model.prompt.parameters()) + list(self.s_model.last.parameters()) + list(self.s_model.kd_last.parameters()) + list(self.s_model.kd_prompt.parameters())
        #     elif self.config['KD_method'] == 'KD' :
        #         s_params_to_opt = list(self.s_model.prompt.parameters()) + list(self.s_model.last.parameters())
        #     elif self.config['KD_method'] == 'DKD' :
        #         s_params_to_opt = list(self.s_model.prompt.parameters()) + list(self.s_model.last.parameters())
        #     elif self.config['KD_method'] == 'FitNets' :
        #         s_params_to_opt = list(self.s_model.prompt.parameters()) + list(self.s_model.last.parameters()) + list(self.s_model.project_t_s.parameters())
        #     elif self.config['KD_method'] == 'ReviewKD' :
        #         s_params_to_opt = list(self.s_model.prompt.parameters()) + list(self.s_model.last.parameters()) + list(self.s_model.ReviewKD_layers.parameters())


        if len(self.config['gpuid']) > 1:
            if self.config['KD_method'] == 'KD_Token':
                if self.s_model.module.prompt_flag == 'apt':
                    print("Usinnnnnnnnnnnnnnng apt")
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        # list(self.s_model.module.clf_norm.parameters()) +
                        list(self.s_model.module.kd_last.parameters()) +
                        list(self.s_model.module.kd_prompt.parameters())
                        # list(self.s_model.module.kd_cls_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        list(self.s_model.module.kd_last.parameters()) +
                        list(self.s_model.module.kd_prompt.parameters())
                    )

            elif self.config['KD_method'] == 'KD':
                if self.s_model.module.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) 
                        # list(self.s_model.module.clf_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters())
                    )

            elif self.config['KD_method'] == 'DKD':
                if self.s_model.module.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) 
                        # list(self.s_model.module.clf_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters())
                    )

            elif self.config['KD_method'] == 'FitNets':
                if self.s_model.module.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        # list(self.s_model.module.clf_norm.parameters()) +
                        list(self.s_model.module.project_t_s.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        list(self.s_model.module.project_t_s.parameters())
                    )

            elif self.config['KD_method'] == 'ReviewKD':
                if self.s_model.module.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        # list(self.s_model.module.clf_norm.parameters()) +
                        list(self.s_model.module.ReviewKD_layers.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.module.prompt.parameters()) +
                        list(self.s_model.module.last.parameters()) +
                        list(self.s_model.module.ReviewKD_layers.parameters())
                    )

        else:
            if self.config['KD_method'] == 'KD_Token':
                if self.s_model.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        # list(self.s_model.clf_norm.parameters()) +
                        list(self.s_model.kd_last.parameters()) +
                        list(self.s_model.kd_prompt.parameters())
                        # list(self.s_model.kd_cls_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        list(self.s_model.kd_last.parameters()) +
                        list(self.s_model.kd_prompt.parameters())
                    )

            elif self.config['KD_method'] == 'KD':
                if self.s_model.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) 
                        # list(self.s_model.clf_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters())
                    )

            elif self.config['KD_method'] == 'DKD':
                if self.s_model.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) 
                        # list(self.s_model.clf_norm.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters())
                    )

            elif self.config['KD_method'] == 'FitNets':
                if self.s_model.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        # list(self.s_model.clf_norm.parameters()) +
                        list(self.s_model.project_t_s.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        list(self.s_model.project_t_s.parameters())
                    )

            elif self.config['KD_method'] == 'ReviewKD':
                if self.s_model.prompt_flag == 'apt':
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        # list(self.s_model.clf_norm.parameters()) +
                        list(self.s_model.ReviewKD_layers.parameters())
                    )
                else:
                    s_params_to_opt = (
                        list(self.s_model.prompt.parameters()) +
                        list(self.s_model.last.parameters()) +
                        list(self.s_model.ReviewKD_layers.parameters())
                    )

       
            
        print('*****************************************')
        s_optimizer_arg = {'params':s_params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            s_optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            s_optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            s_optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            s_optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.s_optimizer = torch.optim.__dict__[self.config['optimizer']](**s_optimizer_arg)
        
        # create schedules
        if self.schedule_type == 'cosine':
            self.s_scheduler = CosineSchedule(self.s_optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.s_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.s_optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self,t_model_name,s_model_name):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.s_model = self.s_model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
            self.s_model = torch.nn.DataParallel(self.s_model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self


class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)

    def create_model(self,t_model_name,s_model_name,shared_para):
        cfg = self.config
        teacher_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param, vit_model=t_model_name, shared_para=shared_para, t_or_s=0)
        student_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param, vit_model=s_model_name, shared_para=shared_para, t_or_s=1)
        return teacher_model, student_model

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self,t_model_name,s_model_name,shared_para):
        cfg = self.config
        t_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual',prompt_param=self.prompt_param, vit_model=t_model_name, shared_para=shared_para, t_or_s=0)
        s_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual',prompt_param=self.prompt_param, vit_model=s_model_name, shared_para=shared_para, t_or_s=1)
        return t_model, s_model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self,t_model_name,s_model_name,shared_para):
        cfg = self.config
        t_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param, vit_model=t_model_name, shared_para=shared_para, t_or_s=0)
        s_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param, vit_model=s_model_name, shared_para=shared_para, t_or_s=1)
        return t_model, s_model
  
########
### Can add more promot_based model
########


class APT(Prompt):
    def __init__(self, learner_config):
        super(APT, self).__init__(learner_config)

    def create_model(self, t_model_name, s_model_name, shared_para):
        cfg = self.config

        teacher_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            out_dim=self.out_dim,
            prompt_flag='apt',
            prompt_param=self.prompt_param,
            vit_model=t_model_name,
            shared_para=shared_para,
            t_or_s=0
        )

        student_model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](
            out_dim=self.out_dim,
            prompt_flag='apt',
            prompt_param=self.prompt_param,
            vit_model=s_model_name,
            shared_para=shared_para,
            t_or_s=1
        )

        return teacher_model, student_model