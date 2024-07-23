import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.models as models
from torch.autograd import Variable
from .vit import VisionTransformer
import numpy as np
import copy

from timm.models.layers import trunc_normal_


import importlib
timm_models = importlib.import_module('timm.models')



class DimensionalityReductionAttention(nn.Module):
    def __init__(self, feature_dim, reduced_dim):
        super().__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, reduced_dim)
        self.scale = 1.0 / (feature_dim ** 0.5)

    def forward(self, x):
        batch_size, num_tokens, feature_dim = x.shape
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Compute the dot products (batch_size, num_tokens, num_tokens)
        attention_scores = torch.bmm(query, key.transpose(1, 2)) * self.scale
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Multiply by values (batch_size, num_tokens, reduced_dim)
        output = torch.bmm(attention_probs, value)

        return output





class CustomNetwork(nn.Module):
    def __init__(self, t_dim, s_dim):
        super(CustomNetwork, self).__init__()
        self.attention_layers = nn.ModuleList([
            DimensionalityReductionAttention(t_dim, s_dim) for _ in range(5)
        ])

    def forward(self, k, v, i):


        k = self.attention_layers[i](k)
        v = self.attention_layers[i](v)

        return k, v



class CodaPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)
        print("n_tasks",n_tasks)
        print("prompt_param:",prompt_param)


        # e prompt init
        for e in self.e_layers:
            # for model saving/loading simplicity, we init the full paramaters here
            # however, please note that we reinit the new components at each task
            # in the "spirit of continual learning", as we don't know how many tasks
            # we will encounter at the start of the task sequence
            #
            # in the original paper, we used ortho init at the start - this modification is more 
            # fair in the spirit of continual learning and has little affect on performance



            e_l = self.e_p_length
            print("!!!!!!!!!self.e_pool_size:",self.e_pool_size)
            print("!!!!!!!!!e_l:",e_l)
            print("!!!!!!!!!emb_d:",emb_d)
            print("!!!!!!!!!self.key_d:",self.key_d)
            p = tensor_prompt(self.e_pool_size, e_l, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            a = tensor_prompt(self.e_pool_size, self.key_d)
            p = self.gram_schmidt(p)
            k = self.gram_schmidt(k)
            a = self.gram_schmidt(a)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    def _init_smart(self, emb_d, prompt_param):

        # prompt basic param
        self.e_pool_size = int(prompt_param[0])
        self.e_p_length = int(prompt_param[1])
        self.e_layers = [0,1,2,3,4]

        # strenth of ortho penalty
        self.ortho_mu = prompt_param[2]
        
    def process_task_count(self):
        self.task_count += 1

        # in the spirit of continual learning, we will reinit the new components
        # for the new task with Gram Schmidt
        #
        # in the original paper, we used ortho init at the start - this modification is more 
        # fair in the spirit of continual learning and has little affect on performance
        # 
        # code for this function is modified from:
        # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
        for e in self.e_layers:
            K = getattr(self,f'e_k_{e}')
            A = getattr(self,f'e_a_{e}')
            P = getattr(self,f'e_p_{e}')
            k = self.gram_schmidt(K)
            a = self.gram_schmidt(A)
            p = self.gram_schmidt(P)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)
            setattr(self, f'e_a_{e}',a)

    # code for this function is modified from:
    # https://github.com/legendongary/pytorch-gram-schmidt/blob/master/gram_schmidt.py
    def gram_schmidt(self, vv):

        def projection(u, v):
            denominator = (u * u).sum()

            if denominator < 1e-8:
                return None
            else:
                return (v * u).sum() / denominator * u

        # check if the tensor is 3D and flatten the last two dimensions if necessary
        is_3d = len(vv.shape) == 3
        if is_3d:
            shape_2d = copy.deepcopy(vv.shape)
            vv = vv.view(vv.shape[0],-1)

        # swap rows and columns
        vv = vv.T

        # process matrix size
        nk = vv.size(1)
        uu = torch.zeros_like(vv, device=vv.device)

        # get starting point
        pt = int(self.e_pool_size / (self.n_tasks))
        s = int(self.task_count * pt)
        f = int((self.task_count + 1) * pt)
        if s > 0:
            uu[:, 0:s] = vv[:, 0:s].clone()
        for k in range(s, f):
            redo = True
            while redo:
                redo = False
                vk = torch.randn_like(vv[:,k]).to(vv.device)
                uk = 0
                for j in range(0, k):
                    if not redo:
                        uj = uu[:, j].clone()
                        proj = projection(uj, vk)
                        if proj is None:
                            redo = True
                            print('restarting!!!')
                        else:
                            uk = uk + proj
                if not redo: uu[:, k] = vk - uk
        for k in range(s, f):
            uk = uu[:, k].clone()
            uu[:, k] = uk / (uk.norm())

        # undo swapping of rows and columns
        uu = uu.T 

        # return from 2D
        if is_3d:
            uu = uu.view(shape_2d)
        
        return torch.nn.Parameter(uu) 

    def forward(self, x_querry, l, x_block=None, train=False, task_id=None, t_or_s = None):

        # e prompts
        e_valid = False

        
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape

            

            K = getattr(self,f'e_k_{l}')
            A = getattr(self,f'e_a_{l}')
            p = getattr(self,f'e_p_{l}')
            pt = int(self.e_pool_size / (self.n_tasks))
            s = int(self.task_count * pt)
            f = int((self.task_count + 1) * pt)
            
            # freeze/control past tasks
            if train:
                if self.task_count > 0:
                    K = torch.cat((K[:s].detach().clone(),K[s:f]), dim=0)
                    A = torch.cat((A[:s].detach().clone(),A[s:f]), dim=0)
                    p = torch.cat((p[:s].detach().clone(),p[s:f]), dim=0)
                else:
                    K = K[s:f]
                    A = A[s:f]
                    p = p[s:f]
            else:
                K = K[0:f]
                A = A[0:f]
                p = p[0:f]
      
            # with attention and cosine sim
            # (b x 1 x d) * soft([1 x k x d]) = (b x k x d) -> attention = k x d
            a_querry = torch.einsum('bd,kd->bkd', x_querry, A)
            # # (b x k x d) - [1 x k x d] = (b x k) -> key = k x d
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(a_querry, dim=2)
            aq_k = torch.einsum('bkd,kd->bk', q, n_K)
            # (b x 1 x k x 1) * [1 x plen x k x d] = (b x plen x d) -> prompt = plen x k x d
            P_ = torch.einsum('bk,kld->bld', aq_k, p)

            # select prompts
            i = int(self.e_p_length/2)
            Ek = P_[:,:i,:]
            Ev = P_[:,i:,:]

            # ortho penalty
            if train and self.ortho_mu > 0:
                loss = ortho_penalty(K) * self.ortho_mu
                loss += ortho_penalty(A) * self.ortho_mu
                loss += ortho_penalty(p.view(p.shape[0], -1)) * self.ortho_mu
            else:
                loss = 0
        else:
            loss = 0

        # combine prompts for prefix tuning
        if e_valid:
            p_return = [Ek, Ev]
        else:
            p_return = None

        # return
        return p_return, loss, x_block

def ortho_penalty(t):
    return ((t @t.T - torch.eye(t.shape[0]).cuda())**2).mean()

# @article{wang2022dualprompt,
#   title={DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Ebrahimi, Sayna and Sun, Ruoxi and Zhang, Han and Lee, Chen-Yu and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and others},
#   journal={European Conference on Computer Vision},
#   year={2022}
# }
class DualPrompt(nn.Module):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim):
        super().__init__()
        self.task_count = 0
        self.emb_d = emb_d
        self.key_d = key_dim
        self.n_tasks = n_tasks
        self._init_smart(emb_d, prompt_param)

        # g prompt init
        for g in self.g_layers:
            p = tensor_prompt(self.g_p_length, emb_d)
            setattr(self, f'g_p_{g}',p)

        # e prompt init
        for e in self.e_layers:
            p = tensor_prompt(self.e_pool_size, self.e_p_length, emb_d)
            k = tensor_prompt(self.e_pool_size, self.key_d)
            setattr(self, f'e_p_{e}',p)
            setattr(self, f'e_k_{e}',k)

    def _init_smart(self, emb_d, prompt_param):
        
        self.top_k = 1
        self.task_id_bootstrap = True

        # prompt locations
        self.g_layers = [0,1]
        self.e_layers = [2,3,4]

        # prompt pool size
        self.g_p_length = int(prompt_param[2])
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

    def process_task_count(self):
        self.task_count += 1

    def forward(self, x_querry, l, x_block, train=False, task_id=None, t_or_s = None):

        # e prompts
        e_valid = False
        if l in self.e_layers:
            e_valid = True
            B, C = x_querry.shape
            K = getattr(self,f'e_k_{l}') # 0 based indexing here
            p = getattr(self,f'e_p_{l}') # 0 based indexing here
            
            # cosine similarity to match keys/querries
            n_K = nn.functional.normalize(K, dim=1)
            q = nn.functional.normalize(x_querry, dim=1).detach()
            cos_sim = torch.einsum('bj,kj->bk', q, n_K)
            
            if train:
                # dual prompt during training uses task id
                if self.task_id_bootstrap:
                    loss = (1.0 - cos_sim[:,task_id]).sum()
                    P_ = p[task_id].expand(len(x_querry),-1,-1)
                else:
                    top_k = torch.topk(cos_sim, self.top_k, dim=1)
                    k_idx = top_k.indices
                    loss = (1.0 - cos_sim[:,k_idx]).sum()
                    P_ = p[k_idx]
            else:
                top_k = torch.topk(cos_sim, self.top_k, dim=1)
                k_idx = top_k.indices
                P_ = p[k_idx]
                
            # select prompts
            if train and self.task_id_bootstrap:
                i = int(self.e_p_length/2)
                Ek = P_[:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,i:,:].reshape((B,-1,self.emb_d))
            else:
                i = int(self.e_p_length/2)
                Ek = P_[:,:,:i,:].reshape((B,-1,self.emb_d))
                Ev = P_[:,:,i:,:].reshape((B,-1,self.emb_d))
        
        # g prompts
        g_valid = False
        if l in self.g_layers:
            g_valid = True
            j = int(self.g_p_length/2)
            p = getattr(self,f'g_p_{l}') # 0 based indexing here
            P_ = p.expand(len(x_querry),-1,-1)
            Gk = P_[:,:j,:]
            Gv = P_[:,j:,:]

        # combine prompts for prefix tuning
        if e_valid and g_valid:
            Pk = torch.cat((Ek, Gk), dim=1)
            Pv = torch.cat((Ev, Gv), dim=1)
            p_return = [Pk, Pv]
        elif e_valid:
            p_return = [Ek, Ev]
        elif g_valid:
            p_return = [Gk, Gv]
            loss = 0
        else:
            p_return = None
            loss = 0

        # return
        if train:
            return p_return, loss, x_block
        else:
            return p_return, 0, x_block

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(DualPrompt):
    def __init__(self, emb_d, n_tasks, prompt_param, key_dim):
        super().__init__(emb_d, n_tasks, prompt_param, key_dim)

    def _init_smart(self, emb_d, prompt_param):
        self.top_k = 5
        self.task_id_bootstrap = False

        # prompt locations
        self.g_layers = []
        if prompt_param[2] > 0:
            self.e_layers = [0,1,2,3,4]
        else:
            self.e_layers = [0]

        # prompt pool size
        self.g_p_length = -1
        self.e_p_length = int(prompt_param[1])
        self.e_pool_size = int(prompt_param[0])

# note - ortho init has not been found to help l2p/dual prompt
def tensor_prompt(a, b, c=None, ortho=False):
    if c is None:
        p = torch.nn.Parameter(torch.FloatTensor(a,b), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(a,b,c), requires_grad=True)
    if ortho:
        nn.init.orthogonal_(p)
    else:
        nn.init.uniform_(p)
    return p    

class ViTZoo(nn.Module):
    def __init__(self, num_classes=10, pt=False, prompt_flag=False, prompt_param=None, vit_model=None, shared_para=None, t_or_s=None):
        super(ViTZoo, self).__init__()

        # get last layer
        self.prompt_flag = prompt_flag
        self.task_id = None
        self.t_or_s = t_or_s
        self.shared_para = shared_para

        if vit_model == 'vit_base_patch16_224':
            embed_dim = 768
            depth = 12
            num_heads = 12

        elif vit_model == 'vit_small_patch16_224':
            embed_dim = 384
            depth = 12
            num_heads = 6
        elif vit_model == 'vit_tiny_patch16_224':
            embed_dim = 192
            depth = 12
            num_heads = 3

        elif vit_model == 'vit_large_patch16_224':
            embed_dim = 1024
            depth = 24
            num_heads = 16
       
        #set knowledge distillation token
        if(t_or_s==1):
            self.kd_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.kd_token, std=.02)
        else:
            self.kd_token = None

        if pt:

            

            s_zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=self.shared_para['s_embed_dim'], depth=self.shared_para['s_depth'],
                                        num_heads=self.shared_para['s_num_heads'], ckpt_layer=0,
                                        drop_path_rate=0
                                        )
            
            s_vit_model_function = getattr(timm_models, self.shared_para['s_vit_name'])
            s_load_dict = s_vit_model_function(pretrained=True).state_dict()

            del s_load_dict['head.weight']; del s_load_dict['head.bias']
            s_zoo_model.load_state_dict(s_load_dict,strict=True)



            zoo_model = VisionTransformer(img_size=224, patch_size=16, embed_dim=embed_dim, depth=depth,
                                        num_heads=num_heads, ckpt_layer=0,
                                        drop_path_rate=0, t_or_s=t_or_s
                                        )
            
            vit_model_function = getattr(timm_models, vit_model)
            load_dict = vit_model_function(pretrained=True).state_dict()

            del load_dict['head.weight']; del load_dict['head.bias']
            zoo_model.load_state_dict(load_dict,strict=True)

 
        #Adding the project_fc_layers for student model

        if(t_or_s==1):
  
            self.project_fc_layers = CodaPrompt(embed_dim, prompt_param[0], prompt_param[1], self.shared_para['t_embed_dim'])
            #self.project_fc_layers = CustomNetwork(768, 192)




        # classifier
        self.last = nn.Linear(embed_dim, num_classes)
        self.kd_last = nn.Linear(embed_dim, num_classes)

        # create prompting module, the teacher model used the prompt with same dimension of student model 
        if self.prompt_flag == 'l2p':
            self.prompt = L2P(embed_dim, prompt_param[0], prompt_param[1], self.shared_para['s_embed_dim'])
        elif self.prompt_flag == 'dual':
            self.prompt = DualPrompt(embed_dim, prompt_param[0], prompt_param[1], self.shared_para['s_embed_dim'])
        elif self.prompt_flag == 'coda':
            self.prompt = CodaPrompt(embed_dim, prompt_param[0], prompt_param[1], self.shared_para['s_embed_dim'])
        else:
            self.prompt = None
        
        # Set feature encoder
        self.feat = zoo_model
        # Teacher model and student model share the query function
        self.s_feat = s_zoo_model

        
    # pen: get penultimate features    
    def forward(self, x, pen=False, train=False, t_p_list_=None, t_corr_list_=None):


        if self.prompt is not None:
            
            #query function
            with torch.no_grad():
                q, _ = self.s_feat(x)
                q = q[:,0,:]

            if(self.t_or_s==0):
                
                out, prompt_loss, p_list_, t_corr_list = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id)
                out = out[:,0,:]
                out = out.view(out.size(0), -1)
                out = self.last(out)
                if self.prompt is not None and train:
                    return out, prompt_loss, p_list_, t_corr_list
                else:
                    return out, p_list_
            
            elif(self.t_or_s==1):

                out, prompt_loss, rm_loss_ = self.feat(x, prompt=self.prompt, q=q, train=train, task_id=self.task_id, t_p_list_ = t_p_list_, project_fc_layers=self.project_fc_layers, t_corr_list_=t_corr_list_)
                
                kd_out = out[:,0,:]
                kd_out = kd_out.view(kd_out.size(0), -1)

                ori_out = out[:,0,:]
                ori_out = ori_out.view(ori_out.size(0), -1)
                

                ori_out = self.last(ori_out)
                kd_out = self.kd_last(kd_out)

                total_out = (ori_out + kd_out)/2

                if self.prompt is not None and train:
                    return ori_out, kd_out, prompt_loss, rm_loss_
                else:
                    return total_out
                
        else:
            
            out, _ = self.feat(x)
            out = out[:,0,:]
            out = out.view(out.size(0), -1)

            if not pen:
                out = self.last(out)
            
            return out


                    




  
            
def vit_pt_imnet(out_dim, block_division = None, prompt_flag = 'None', prompt_param=None, vit_model=None, shared_para=None, t_or_s=None):
    return ViTZoo(num_classes=out_dim, pt=True, prompt_flag=prompt_flag, prompt_param=prompt_param, vit_model=vit_model, shared_para=shared_para, t_or_s=t_or_s)

