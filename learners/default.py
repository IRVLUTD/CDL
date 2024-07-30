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
import copy
import numpy as np
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
            s_acc = AverageMeter()

            for epoch in range(self.config['schedule'][-1]):
                self.epoch=epoch

                if epoch > 0: self.scheduler.step()
                for param_group in self.optimizer.param_groups:
                    self.log('LR:', param_group['lr'])
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
                    s_loss, soft_loss, s_output= self.s_update_model(x, y, cur_logits, p_list_, t_corr_list_)


                    
                    # measure accuracy and record loss
                    y = y.detach()
                    accumulate_acc(output, y, task, acc, topk=(self.top_k,))
                    accumulate_acc(s_output, y, task, s_acc, topk=(self.top_k,))
                    losses.update(loss,  y.size(0))
                    s_losses.update(s_loss,  y.size(0))
                    soft_losses.update(soft_loss,  y.size(0))


                # eval update
                self.log('Epoch:{epoch:.0f}/{total:.0f}'.format(epoch=self.epoch+1,total=self.config['schedule'][-1]))
                self.log(' * Loss {loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=losses,acc=acc))

                self.log(' * Student_Loss {loss.avg:.3f} | Soft_Loss {soft_loss.avg:.3f} | Train Acc {acc.avg:.3f}'.format(loss=s_losses, soft_loss=soft_losses, acc=s_acc))

                # reset
                losses = AverageMeter()
                acc = AverageMeter()

                s_losses = AverageMeter()
                soft_losses = AverageMeter()
                s_acc = AverageMeter()
                
        self.model.eval()
        self.s_model.eval()

        self.last_valid_out_dim = self.valid_out_dim
        self.first_task = False

        # Extend memory
        self.task_count += 1
        if self.memory_size > 0:
            train_dataset.update_coreset(self.memory_size, np.arange(self.last_valid_out_dim))


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
    
    def coda_get_t_p_list_(self, input, s_feat, K_list, A_list, P_list):

        with torch.no_grad():
            x_querry, _ = s_feat(input)
            x_querry = x_querry[:,0,:]

        prompt_param = self.config['prompt_param']
        e_pool_size = int(prompt_param[1][0])   
        n_tasks = int(prompt_param[0])
        e_p_length = int(prompt_param[1][1])

        task_count = self.model.module.prompt.task_count

        e_layers = [0,1,2,3,4]
        p_list_ = []

        pt = int(e_pool_size / (n_tasks))

        f = int((task_count + 1) * pt)

        e_index = 0
        for l in range(int(self.shared_para['t_depth'])):
            if l in e_layers:
                K = K_list[e_index][0:f]
                A = A_list[e_index][0:f]
                P = P_list[e_index][0:f]
                e_index = e_index + 1
            
                a_querry = torch.einsum('bd,kd->bkd', x_querry, A)       
                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(a_querry, dim=2)
                aq_k = torch.einsum('bkd,kd->bk', q, n_K)
                P_ = torch.einsum('bk,kld->bld', aq_k, P)
                i = int(e_p_length/2)
                Ek = P_[:,:i,:]
                Ev = P_[:,i:,:]

                p_return = [Ek, Ev]
            else:
                p_return = None
            p_list_.append(p_return)
        return p_list_
                
    
    def dual_get_t_p_list_(self, input, s_feat, E_K_list, E_P_list, G_P_list):

        with torch.no_grad():
            x_querry, _ = s_feat(input)
            x_querry = x_querry[:,0,:]
        B, C = x_querry.shape
        prompt_param = self.config['prompt_param']
        n_tasks = int(prompt_param[0])
        g_p_length = int(prompt_param[1][2])
        e_p_length = int(prompt_param[1][1])
        e_pool_size = int(prompt_param[1][0])

        g_layers = [0,1]
        e_layers = [2,3,4]
        topk = 1
        p_list_ = []

        e_index = 0
        g_index = 0
        for l in range(int(self.shared_para['t_depth'])):
            e_valid = False
            if l in e_layers:
                e_valid = True
                K = E_K_list[e_index]
                P = E_P_list[e_index]
                e_index = e_index + 1

                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1).detach()
                cos_sim = torch.einsum('bj,kj->bk', q, n_K)

                top_k = torch.topk(cos_sim, topk, dim=1)
                k_idx = top_k.indices
                P_ = P[k_idx]

                i = int(e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.shared_para['t_embed_dim']))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.shared_para['t_embed_dim']))
            
            g_valid = False
            if l in g_layers:
                g_valid = True
                j = int(g_p_length/2)
                P = G_P_list[g_index]
                g_index = g_index + 1

                P_ = P.expand(len(x_querry),-1,-1)
                Gk = P_[:,:j,:]
                Gv = P_[:,j:,:]
            
            if e_valid and g_valid:
                Pk = torch.cat((Ek, Gk), dim=1)
                Pv = torch.cat((Ev, Gv), dim=1)
                p_return = [Pk, Pv]
            elif e_valid:
                p_return = [Ek, Ev]
            elif g_valid:
                p_return = [Gk, Gv]
            else:
                p_return = None
            
            p_list_.append(p_return)
        
        return p_list_
            
    def l2p_get_t_p_list_(self, input, s_feat, E_K_list, E_P_list, G_P_list):
        
        with torch.no_grad():
            x_querry, _ = s_feat(input)
            x_querry = x_querry[:,0,:]
        B, C = x_querry.shape
        prompt_param = self.config['prompt_param']
        n_tasks = int(prompt_param[0])
        g_p_length = -1
        e_p_length = int(prompt_param[1][1])
        e_pool_size = int(prompt_param[1][0])

        g_layers = []
        e_layers = [0]
        topk = 5
        p_list_ = []

        e_index = 0
        g_index = 0        
        for l in range(int(self.shared_para['t_depth'])):
            e_valid = False
            if l in e_layers:
                e_valid = True
                K = E_K_list[e_index]
                P = E_P_list[e_index]
                e_index = e_index + 1

                n_K = nn.functional.normalize(K, dim=1)
                q = nn.functional.normalize(x_querry, dim=1).detach()
                cos_sim = torch.einsum('bj,kj->bk', q, n_K)

                top_k = torch.topk(cos_sim, topk, dim=1)
                k_idx = top_k.indices
                P_ = P[k_idx]

                i = int(e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.shared_para['t_embed_dim']))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.shared_para['t_embed_dim']))
            
            g_valid = False
            if l in g_layers:
                g_valid = True
                j = int(g_p_length/2)
                P = G_P_list[g_index]
                g_index = g_index + 1

                P_ = P.expand(len(x_querry),-1,-1)
                Gk = P_[:,:j,:]
                Gv = P_[:,j:,:]
            
            if e_valid and g_valid:
                Pk = torch.cat((Ek, Gk), dim=1)
                Pv = torch.cat((Ev, Gv), dim=1)
                p_return = [Pk, Pv]
            elif e_valid:
                p_return = [Ek, Ev]
            elif g_valid:
                p_return = [Gk, Gv]
            else:
                p_return = None
            
            p_list_.append(p_return)
        
        return p_list_

    ########
    ### Can add more promot_based model
    ########



    def validation(self, dataloader, model=None, task_metric='acc',  verbal = True, task_global=False, t_or_s=None, t_p_list_=None):

        if model is None:
            model = self.model
            s_model = self.s_model

        acc = AverageMeter()
        s_acc = AverageMeter()
        orig_mode = model.training
        model.eval()

        s_orig_mode = s_model.training
        s_model.eval()

        if(self.config['learner_name'] == 'CODAPrompt'):
            K_list, A_list, P_list = model.module.prompt.get_K_A_P()

        elif(self.config['learner_name'] == 'DualPrompt'):
            E_K_list, E_P_list, G_P_list = model.module.prompt.get_EK_EP_GP()

        elif(self.config['learner_name'] == 'L2P'):
            L2P_E_K_list, L2P_E_P_list, L2P_G_P_list = model.module.prompt.get_EK_EP_GP()
        
        ########
        ### Can add more promot_based model
        ########

        s_feat = model.module.s_feat

        for i, (input, target, task) in enumerate(dataloader):

            if self.gpu:
                with torch.no_grad():
                    input = input.cuda()
                    target = target.cuda()

            output, _ = model.forward(input)
            # output = output[:, :self.valid_out_dim]
            if(self.config['learner_name'] == 'CODAPrompt'):
                p_list_test = self.coda_get_t_p_list_(input, s_feat, K_list, A_list, P_list)
            elif(self.config['learner_name'] == 'DualPrompt'):
                p_list_test = self.dual_get_t_p_list_(input, s_feat, E_K_list, E_P_list, G_P_list)
            elif(self.config['learner_name'] == 'L2P'):
                p_list_test = self.l2p_get_t_p_list_(input, s_feat, L2P_E_K_list, L2P_E_P_list, L2P_G_P_list)

            ########
            ### Can add more promot_based model
            ########
 
            s_output = s_model.forward(input, t_p_list_ = p_list_test)[:, :self.valid_out_dim]

            #Calculate the accuracy
            acc = accumulate_acc(output, target, task, acc, topk=(self.top_k,))
            s_acc = accumulate_acc(s_output, target, task, s_acc, topk=(self.top_k,))
          
            
        model.train(orig_mode)
        s_model.train(s_orig_mode)

        if verbal:
            self.log(' * Val Teacher Acc {acc.avg:.3f}'
                    .format(acc=acc))
            
            self.log(' * Val Student Acc {acc.avg:.3f}'
                    .format(acc=s_acc))
        

        return acc.avg, s_acc.avg
            

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
        #Saving teacher model
        model_state = self.model.state_dict()
        for key in model_state.keys(): 
            model_state[key] = model_state[key].cpu()
        self.log('=> Saving Teacher_class model to:', filename)
        torch.save(model_state, filename + 'T_class.pth')
        self.log('=> Save Done')

        #Saving student model
        s_model_state = self.s_model.state_dict()
        for key in s_model_state.keys(): 
            s_model_state[key] = s_model_state[key].cpu()
        self.log('=> Saving Student_class model to:', filename)
        torch.save(s_model_state, filename + 'S_class.pth')
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