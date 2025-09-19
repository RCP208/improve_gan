# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# modified by Axel Sauer for "Projected GANs Converge Faster"
#
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
import torch.nn as nn
from torch_utils.random_queue import Random_queue, Random_queue_torch # PyTorch version of queue
from torch_utils.data_queue import Data_queue
from training import hellinger_loss

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()
grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook
        

class ProjectedGANLoss(Loss):
    def __init__(self, device, G, D, G_ema, blur_init_sigma=0, blur_fade_kimg=0, clc_loss_type='square', clc_loss_weight = 1,factors=100, batch_size_queue=64,**kwargs):
        super().__init__()
        self.device = device
        self.G = G
        self.G_ema = G_ema
        self.D = D
        self.blur_init_sigma = blur_init_sigma
        self.blur_fade_kimg = blur_fade_kimg

        # 误差积分
        self.g_queue = Data_queue(capacity=3)
        self.d_queue = Data_queue(capacity=3)
        
        # clc loss parameters 
        if clc_loss_weight != 0:
            self.i_real_queue = Random_queue_torch(batch_size_queue *factors,batch_size_queue) # queue for real images
            self.i_fake_queue = Random_queue_torch(batch_size_queue *factors,batch_size_queue) # queue for fake images
            self.clc_loss_type = clc_loss_type # loss type for clc loss
            self.clc_loss_weight = clc_loss_weight # loss weight for clc loss
            self.max0 = torch.nn.ReLU()  # max(0,x)

    def run_G(self, z, c, update_emas=False):
        ws = self.G.mapping(z, c, update_emas=update_emas)
        img = self.G.synthesis(ws, c, update_emas=False)
        return img

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())

        logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        if phase in ['Dreg', 'Greg']: return  # no regularization needed for PG
        # blurring schedule
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 1 else 0

        if do_Gmain:

            # Gmain: Maximize logits for generated images.
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img = self.run_G(gen_z, gen_c)
                gf, gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)

                real_img_tmp = real_img.detach().requires_grad_(False)
                rf, real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)

                # g_c = [-((F.relu(torch.ones_like(dr) - dr)).mean() + (F.relu(torch.ones_like(df) + df)).mean()) for (dr, df) in zip(real_logits[0], gen_logits[0])]
                # softmax = torch.nn.Softmax(dim=0)
                # coe = softmax(torch.tensor(g_c))

                mse = nn.MSELoss()

                k2 = 2.5#max(2.5 * (1-cur_nimg / 1.3e7), 0.1)
                content_list = []
                for i in range(4):
                    content_list.append(mse(rf[str(i)], gf[str(i)]))
                gloss_content = sum(content_list) * 0.01 #* max((1 - cur_nimg / 1.3e7), 0)
                # real_img_tmp = real_img.detach().requires_grad_(False)

                #
                cur_mse_loss = mse(gen_img, real_img_tmp) * k2
                # k2 = max(2.5 * (1-cur_nimg / 1.3e7), 0)
                # flag = abs(real_img_tmp.mean() - gen_img.mean()) / real_img_tmp.mean()
                # if flag > 0.05:
                #     loss_Gmain = sum([(-l).mean() for l in gen_logits]) + mse_loss * k1
                # else:

                loss_Gmain = sum([(-l).mean() for l in (gen_logits[0])])
                gen_logits = torch.cat(gen_logits[0], dim=1).view(-1)
                # M_G = abs(real_img_tmp.mean() - gen_img.mean())

                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                training_stats.report('Loss/G/cont', gloss_content)
                training_stats.report('Loss/G/mse', cur_mse_loss)
                # training_stats.report('Loss/G/cur_mse', cur_mse_loss)
                # training_stats.report('Loss/G/M_G', M_G)
                # training_stats.report('Loss/G/adv_loss', adv_gloss)
                training_stats.report('Loss/G/loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain).backward()

        if do_Dmain:
            
            # Dmain: Minimize logits for generated images.
            with torch.autograd.profiler.record_function('D_forward'):
                gen_img = self.run_G(gen_z, gen_c, update_emas=True)
                gf, gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)

                fake_var, fake_mean = var_mean_loss(gen_logits[0])
                # loss_Dgen = sum([(F.relu(torch.ones_like(l) + l)).mean() for l in gen_logits])
                # gen_logits = torch.cat(gen_logits)
                
                # Logging
                # training_stats.report('Loss/scores/fake', gen_logits)
                # training_stats.report('Loss/signs/fake', gen_logits.sign())

            # with torch.autograd.profiler.record_function('Dgen_backward'):
            #     loss_Dgen.backward()
            
            # Dmain: Maximize logits for real images.
            # with torch.autograd.profiler.record_function('Dreal_forward'):
                real_img_tmp = real_img.detach().requires_grad_(False)
                rf, real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                real_var, real_mean = var_mean_loss(real_logits[0])
                # loss_Dreal = sum([(F.relu(torch.ones_like(l) - l)).mean() for l in real_logits])
                # real_logits = torch.cat(real_logits)
                d_c = [((F.relu(torch.ones_like(dr) - dr)).mean() + (F.relu(torch.ones_like(df) + df)).mean()) for (dr, df) in zip(real_logits[0], gen_logits[0])]
                softmax = torch.nn.Softmax(dim=0)
                coe = softmax(torch.tensor(d_c))
                loss_Dgen_individual = [(F.relu(torch.ones_like(l) + l)).mean() for c, l in zip(coe, gen_logits[0])]
                loss_Dreal_individual = [(F.relu(torch.ones_like(l) - l)).mean() for c, l in zip(coe, real_logits[0])]
                loss_Dgen = sum(loss_Dgen_individual)
                loss_Dreal = sum(loss_Dreal_individual)


                gen_logits = torch.cat(gen_logits[0], dim=1).view(-1) # batch * 4 * 100
                real_logits = torch.cat(real_logits[0], dim=1).view(-1)


                # Logging
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())
                training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                training_stats.report('Loss/D/loss_gen/1', loss_Dgen_individual[0])
                training_stats.report('Loss/D/loss_gen/2', loss_Dgen_individual[1])
                training_stats.report('Loss/D/loss_gen/3', loss_Dgen_individual[2])
                training_stats.report('Loss/D/loss_gen/4', loss_Dgen_individual[3])
                training_stats.report('Loss/D/loss_real/1', loss_Dreal_individual[0])
                training_stats.report('Loss/D/loss_real/2', loss_Dreal_individual[1])
                training_stats.report('Loss/D/loss_real/3', loss_Dreal_individual[2])
                training_stats.report('Loss/D/loss_real/4', loss_Dreal_individual[3])

                training_stats.report('Loss/scores/DI', real_logits.min() - gen_logits.max()) # DI score
                training_stats.report('Loss/scores/W', real_logits.mean() - gen_logits.mean()) # W distance
                training_stats.report('Loss/scores/D', real_img_tmp.mean() - gen_img.mean()) # W distance
                training_stats.report('Loss/fake_var/D', fake_var) # W distance
                training_stats.report('Loss/fake_mean/D', fake_mean) # W distance
                training_stats.report('Loss/real_var/D', real_var) # W distance
                training_stats.report('Loss/real_mean/D', real_mean) # W distance

            with torch.autograd.profiler.record_function('D_backward'):
                (loss_Dreal + loss_Dgen).backward()
                
            if self.clc_loss_weight > 0:
                with torch.autograd.profiler.record_function('Dclc_forward'):
                    # save old images to queue
                    self.i_real_queue.set_data(real_img)
                    self.i_fake_queue.set_data(gen_img)

                    # get iamges from queue
                    i_xreal = self.i_real_queue.get_data()
                    i_xfake = self.i_fake_queue.get_data()

                    # remove tensor to the real device
                    i_xreal = i_xreal.cuda(device=real_img.device)
                    i_xfake = i_xfake.cuda(device=gen_img.device)

                    # Mix Up function with real image and generated image, this is not used in our work
                    # lam = np.random.beta(0.4, 0.4)
                    # i_xreal = lam * i_xreal + (1-lam) * i_xreal_2
                    # i_xfake = lam * i_xfake + (1-lam) * i_xfake_2

                    rf, i_real_doutput = self.run_D(i_xreal, real_c, blur_sigma=blur_sigma)
                    gf, i_fake_doutput = self.run_D(i_xfake, gen_c, blur_sigma=blur_sigma)
                    # i_real_doutput = self.run_D(real_img, real_c, blur_sigma=blur_sigma)

                    if self.clc_loss_type == 'square':
                        # version of Type 'square', with `(D(P(x)))^2 + (D(P(G(z))))^2`
                        reg =( sum([(l**2).mean() for l in i_fake_doutput[0]]) +
                                sum([(l**2).mean() for l in i_real_doutput[0]]))
                        index = 1 / (i_xreal.mean() ** 2 + i_xfake.mean() ** 2)
                        k = max(0.1 * (1-cur_nimg / 2.6e7), 0.01)
                        M_D = (real_img_tmp.mean() + gen_img.mean())
                        M_D2  = abs(real_img_tmp.mean() - gen_img.mean()) / (real_img_tmp.mean() + gen_img.mean())
                        k1 = max(k - M_D2, 0.1)
                        # reg = sum([((l - 1)**2).mean() for l in i_real_doutput])
                        i_loss = reg * k
                        # i_loss = reg * self.clc_loss_weight
                    elif self.clc_loss_type == 'abs':
                        # version of Type 'abs', with `D(P(x)) + D(P(G(z)))`
                        i_loss =( sum([(torch.abs(l)).mean() for l in i_fake_doutput]) +
                                sum([(torch.abs(l)).mean() for l in i_real_doutput]) ) * self.clc_loss_weight
                        
                    training_stats.report('Loss/D/i_loss', i_loss)
                    training_stats.report('Loss/D/reg', reg)
                    training_stats.report('Loss/D/M_D', M_D)
                    training_stats.report('Loss/D/M_D2', M_D2)
                    training_stats.report('Loss/D/index', index)
                # with torch.autograd.profiler.record_function('Dclc_backward'):
                #     i_loss.backward()

def var_mean_loss(dout):
    var_outer = []
    mean_outer = []
    for out in dout:
        var = torch.var(out, dim=1)
        mean = torch.mean(out)
        var_outer.append(max(var))
        mean_outer.append(mean)
    var_loss = sum(var_outer)
    mean_loss = sum(mean_outer)

    return var_loss, mean_loss

# def min_dist_inlist(input):
#     if len(input) == 1:
#         return input[0]
#     input.sort()
#     dist = []
#     for i in range(len(input)):
#         if i < (len(input) - 1):
#             dist.append(input[i + 1] - input[i])
#     return max(dist)



