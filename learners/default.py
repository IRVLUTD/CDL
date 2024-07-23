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
import copy
from utils.schedulers import CosineSchedule

class NormalNN(nn.Module):
    '''
    Normal Neural Network with SGD for classification
    '''
    def __init__(self, learner_config):

        super(NormalNN, self).__init__()
        self.log = print
        self.config = learner_config
        self.out_dim = learner_config['out_dim']
        self.t_model_name = learner_config['t_model']
        self.s_model_name = learner_config['s_model']

        if self.s_model_name == 'vit_base_patch16_224':
            s_embed_dim = 768
            s_depth = 12
            s_num_heads = 12

        elif self.s_model_name == 'vit_small_patch16_224':
            s_embed_dim = 384
            s_depth = 12
            s_num_heads = 6
        elif self.s_model_name == 'vit_tiny_patch16_224':
            s_embed_dim = 192
            s_depth = 12
            s_num_heads = 3

        elif self.s_model_name == 'vit_large_patch16_224':
            s_embed_dim = 1024
            s_depth = 24
            s_num_heads = 16


        if self.t_model_name == 'vit_base_patch16_224':
            t_embed_dim = 768
            t_depth = 12
            t_num_heads = 12

        elif self.t_model_name == 'vit_small_patch16_224':
            t_embed_dim = 384
            t_depth = 12
            t_num_heads = 6
        elif self.t_model_name == 'vit_tiny_patch16_224':
            t_embed_dim = 192
            t_depth = 12
            t_num_heads = 3

        elif self.t_model_name == 'vit_large_patch16_224':
            t_embed_dim = 1024
            t_depth = 24
            t_num_heads = 16
       

        self.shared_para = {'t_vit_name': self.t_model_name,
                        't_embed_dim': t_embed_dim,
                        't_depth': t_depth,
                        't_num_heads': t_num_heads,
                        's_vit_name': self.s_model_name,
                        's_embed_dim': s_embed_dim,
                        's_depth': s_depth,
                        's_num_heads': s_num_heads,
                        }

        self.model, self.s_model = self.create_model(self.t_model_name,self.s_model_name,self.shared_para)
        self.reset_optimizer = True
        self.overwrite = learner_config['overwrite']
        self.batch_size = learner_config['batch_size']
        self.tasks = learner_config['tasks']
        self.top_k = learner_config['top_k']

        # replay memory parameters
        self.memory_size = self.config['memory']
        self.task_count = 0

        # class balancing
        self.dw = self.config['DW']
        if self.memory_size <= 0:
            self.dw = False

        # supervised criterion
        self.criterion_fn = nn.CrossEntropyLoss(reduction='none')
        
        # cuda gpu
        if learner_config['gpuid'][0] >= 0:
            self.cuda()
            self.gpu = True
        else:
            self.gpu = False
        
        # highest class index from past task
        self.last_valid_out_dim = 0 

        # highest class index from current task
        self.valid_out_dim = 0

        # set up schedules
        self.schedule_type = self.config['schedule_type']
        self.schedule = self.config['schedule']

        # initialize optimizer
        self.init_optimizer()



    ##########################################
    #           MODEL TRAINING               #
    ##########################################

    def learn_batch(self, train_loader, train_dataset, model_save_dir, args, val_loader=None):
        
        # try to load model
        need_train = True
        if not self.overwrite:
            try:
                self.load_model(model_save_dir)
                need_train = False
            except:
                pass

        # trains
        if self.reset_optimizer:  # Reset optimizer before learning each task
            self.log('Optimizer is reset!')
            self.init_optimizer()
        if need_train:
            
            # data weighting
            self.data_weighting(train_dataset)
            losses = AverageMeter()
            acc = AverageMeter()

            s_losses = AverageMeter()
            soft_losses = AverageMeter()
            rm_losses = AverageMeter()
            s_acc = AverageMeter()

            batch_time = AverageMeter()
            batch_timer = Timer()
            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
                batch_timer.tic()
                for i, (x, y, task)  in enumerate(train_loader):

                    # verify in train mode
                    self.model.train()
                    self.s_model.train()

                    # send data to gpu
                    if self.gpu:
                        x = x.cuda()
                        y = y.cuda()
                    
                    # model update
                    loss, output, cur_logits, p_list_, t_corr_list_= self.update_model(x, y, epoch)        
                    # pre_cls_logits = torch.softmax(output,dim=-1)
                    # predicts_ = torch.max(pre_cls_logits, dim=1)[1]
                    s_loss, soft_loss, rm_loss_, s_output= self.s_update_model(x, y, cur_logits, p_list_, t_corr_list_)

                    # measure elapsed time
                    batch_time.update(batch_timer.toc())  
                    batch_timer.tic()
                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    accumulate_acc(s_output, y, task, s_acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0))
                    s_losses.update(s_loss,  y.size(0))
                    soft_losses.update(soft_loss,  y.size(0))
                    rm_losses.update(rm_loss_, y.size(0))
                    batch_timer.tic()

                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                self.log(' * Student_Loss {loss.avg:.3f} | Soft_Loss {soft_loss.avg:.3f} | Rm_Loss {rm_loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=s_losses, soft_loss=soft_losses, rm_loss=rm_losses, acc=s_acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                s_losses = AverageMeter()
                soft_losses = AverageMeter()
                rm_losses = AverageMeter()
                s_acc = AverageMeter()
                
        self.model.eval()
        self.s_model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))

        try:
            return batch_time.avg
        except:
            return None

    def criterion(self, logits, targets, data_weights):
        loss_supervised = (self.criterion_fn(logits, targets.long()) * data_weights).mean()
        return loss_supervised 

    def update_model(self, inputs, targets, target_scores = None, dw_force = None, kd_index = None):

        
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        logits = self.forward(inputs)
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits
    


    def validation(self, dataloader, model=None, task_in = None, task_metric='acc',  verbal = True, task_global=False, t_or_s=None, t_p_list_=None):

        if model is None:
            # if t_or_s == 0:
            #     model = self.model
            # elif t_or_s ==1:
            #     model = self.s_model

            model = self.model
            s_model = self.s_model

        # This function doesn't distinguish tasks.
        batch_timer = Timer()
        acc = AverageMeter()
        s_acc = AverageMeter()
        batch_timer.tic()

        orig_mode = model.training
        model.eval()

        s_orig_mode = s_model.training
        s_model.eval()

        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()
            if task_in is None:
                output, p_list_ = model.forward(input)
                output = output[:, :self.valid_out_dim]
                s_output = s_model.forward(input, t_p_list_ = p_list_)[:, :self.valid_out_dim]

                # self.log(' * Target {target}, Task {task}'
                #     .format(target=target, task=task))
                acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                s_acc = accumulate_acc(s_output, target, task, s_acc, topk=(self.top_k,))
            else:
                mask = target >= task_in[0]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]

                mask = target < task_in[-1]
                mask_ind = mask.nonzero().view(-1) 
                input, target = input[mask_ind], target[mask_ind]
                
                if len(target) > 1:
                    if task_global:
                        #output = model.forward(input)[:, :self.valid_out_dim]
                        output, p_list_ = model.forward(input)
                        output = output[:, :self.valid_out_dim]
                        s_output = s_model.forward(input, t_p_list_ = p_list_)[:, :self.valid_out_dim]

                        acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
                        s_acc = accumulate_acc(s_output, target, task, s_acc, topk=(self.top_k,))
                    else:
                        #output = model.forward(input)[:, task_in]
                        output, p_list_ = model.forward(input)
                        output = output[:, task_in]
                        s_output = s_model.forward(input, t_p_list_ = p_list_)[:, task_in]
                        acc = accumulate_acc(output, target-task_in[0], task, acc, topk=(self.top_k,))
                        s_acc = accumulate_acc(s_output, target-task_in[0], task, s_acc, topk=(self.top_k,))
            
        model.train(orig_mode)
        s_model.train(s_orig_mode)

        if verbal:
            self.log(' * Val Teacher Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=acc, time=batch_timer.toc()))
            
            self.log(' * Val Student Acc {acc.avg:.3f}, Total time {time:.2f}'
                    .format(acc=s_acc, time=batch_timer.toc()))
        

        return acc.avg, s_acc.avg
            
        # if(t_or_s == 0):
        #     return acc.avg, p_list_
        # elif(t_or_s == 1):
        #     return acc.avg

    ##########################################
    #             MODEL UTILS                #
    ##########################################

    # data weighting
    def data_weighting(self, dataset, num_seen=None):
        self.dw_k = torch.tensor(np.ones(self.valid_out_dim + 1, dtype=np.float32))
        # cuda
        if self.cuda:
            self.dw_k = self.dw_k.cuda()

    def save_model(self, filename):
        model_state = self.model.state_dict()
        for key in model_state.keys():  # Always save it to cpu
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving class model to:', filename)
        torch.save(model_state, filename + 'class.pth')
        self.log('=> Save Done')

    def load_model(self, filename):
        self.model.load_state_dict(torch.load(filename + 'class.pth'))
        self.log('=> Load Done')
        if self.gpu:
            self.model = self.model.cuda()
        self.model.eval()

    def load_model_other(self, filename, model):
        model.load_state_dict(torch.load(filename + 'class.pth'))
        if self.gpu:
            model = model.cuda()
        return model.eval()

    # sets model optimizers
    def init_optimizer(self):
        
 

        # parse optimizer args
        optimizer_arg = {'params':self.model.parameters(),
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


        

  

 

    def print_model(self):
        self.log(self.model)
        self.log('#parameter of model:', self.count_parameter())
    
    def reset_model(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        return self.model.forward(x)[:, :self.valid_out_dim]


    def predict(self, inputs):
        self.model.eval()
        out = self.forward(inputs)
        return out
    
    def add_valid_output_dim(self, dim=0):
        # This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
        self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
        self.valid_out_dim += dim
        self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
        return self.valid_out_dim

    def count_parameter(self):
        return sum(p.numel() for p in self.model.parameters())   

    def count_memory(self, dataset_size):
        return self.count_parameter() + self.memory_size * dataset_size[0]*dataset_size[1]*dataset_size[2]

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])
        self.model = self.model.cuda()
        self.criterion_fn = self.criterion_fn.cuda()
        # Multi-GPU
        if len(self.config['gpuid']) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.log("Running on:", device)
        return device

    def pre_steps(self):
        pass

class FinetunePlus(NormalNN):

    def __init__(self, learner_config):
        super(FinetunePlus, self).__init__(learner_config)

    def update_model(self, inputs, targets, target_KD = None):

        # get output
        logits = self.forward(inputs)

        # standard ce
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets.long(), dw_cls)

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        return total_loss.detach(), logits

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def accumulate_acc(output, target, task, meter, topk):
    meter.update(accuracy(output, target, topk), len(target))
    return meter