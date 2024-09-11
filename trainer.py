import torch
import utility
from decimal import Decimal
from tqdm import tqdm

import torch.nn as nn
import torch
import math
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from torch.nn.parameter import Parameter
from torch.autograd import Variable
# from IPython import embed

import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

class Trainer():
    def __init__(self, opt, loader, my_model, my_loss, ckp):
        self.opt = opt
        self.scale = opt.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(opt, self.model)
        self.scheduler = utility.make_scheduler(opt, self.optimizer)
        self.error_last = 1e8

    def train(self):
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)#GPU
            timer_data.hold()
            timer_model.tic()
            
            self.optimizer.zero_grad()

            sr = self.model(lr[0])
            loss = self.loss(sr, hr)
            
            if loss.item() < self.opt.skip_threshold * self.error_last:
                loss.backward()                
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
                
            timer_model.hold()

            if (batch + 1) % self.opt.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.opt.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.step()


    def test(self):
        epoch = self.scheduler.last_epoch
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, 1))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            scale = max(self.scale)
            for si, s in enumerate([scale]):
                eval_psnr = 0
                eval_ssim = 0
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for _, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    
                    with torch.no_grad():
                        # sr = forward_chop(self.model, lr[0], scale)  ## lwj
                        sr = self.model(lr[0])
                        # print("lr", lr[0].size())
                        print("sr", sr.size())
                    sr_another = sr  ## add calculate ssim by lwj
                    if isinstance(sr, list): sr = sr[-1]

                    sr = utility.quantize(sr, self.opt.rgb_range)
                    
                    ############ add calculate ssim by lwj ###########
                    hr_another = hr.clamp(0, 255)
                    print("hr_another", hr_another.size())
                    sr_another = sr
                    if isinstance(sr, list): sr = sr[-1]
                    sr_another = sr_another.clamp(0, 255)
                    hr_ycbcr = utility.rgb_to_ycbcr(hr_another)
                    sr_ycbcr = utility.rgb_to_ycbcr(sr_another)
                    hr_another = hr_ycbcr[:, 0:1, :, :]
                    sr_another = sr_ycbcr[:, 0:1, :, :]
                    hr_another = hr_another[:, :, scale:-scale, scale:-scale]
                    sr_another = sr_another[:, :, scale:-scale, scale:-scale]
                    eval_ssim += utility.calc_ssim(sr_another, hr_another)
                    ############ add calculate ssim by lwj ###########
                    

                    if not no_eval:
                        eval_psnr += utility.calc_psnr(
                            sr, hr, s, self.opt.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        # eval_psnr += utility.calc_psnr1(sr_another, hr_another)  
                        ############ add calculate ssim by lwj ###########
                        # eval_ssim += utility.calc_ssim(sr_another, hr_another)

                    # save test results
                    if self.opt.save_results:
                        self.ckp.save_results_nopostfix(filename, sr, s)

                self.ckp.log[-1, si] = (eval_psnr / len(self.loader_test))
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f}  SSIM: {:.5f} (Best: {:.3f} @epoch {})'.format(
                        self.opt.data_test, s,
                        self.ckp.log[-1, si],
                        eval_ssim / len(self.loader_test),
                        best[0][si],
                        best[1][si] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.opt.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def step(self):
        self.scheduler.step()

    def prepare(self, *args):
        device = torch.device('cpu' if self.opt.cpu else 'cuda')
            
        if len(args)>1:
            return [a.to(device) for a in args[0]], args[-1].to(device)
        return [a.to(device) for a in args[0]], 

    def terminate(self):
        if self.opt.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch
            return epoch >= self.opt.epochs
