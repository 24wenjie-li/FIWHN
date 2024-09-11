import math
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

from torch.autograd import Variable

def mean_channels(x):
    assert(x.dim() == 4)
    spatial_sum = x.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (x.shape[2] * x.shape[3])

def std(x):
    assert(x.dim() == 4)
    x_mean = mean_channels(x)
    x_var = (x - x_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (x.shape[2] * x.shape[3])
    return x_var.pow(0.5)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def default_conv_stride2(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,stride=2,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class CEALayer(nn.Module):
    def __init__(self, n_feats=64, reduction=16):
        super(CEALayer, self).__init__()
        self.conv_du = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 5, padding=1, groups=n_feats // reduction, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class MRCAB(nn.Module):
    def __init__(self,n_feat,reduction=16,act=nn.ReLU(True), res_scale=1,t=2):
        super(MRCAB, self).__init__()
        self.ca = CALayer(n_feat, reduction)
        self.relu = act
        self.depth_conv = nn.Conv2d(in_channels=n_feat*t,out_channels=n_feat*t,kernel_size=3,stride=1,padding=1,groups=n_feat*t)
        self.point_conv1 = nn.Conv2d(in_channels=n_feat,out_channels=n_feat*t,kernel_size=1,padding=0,groups=1)
        self.point_conv2 = nn.Conv2d(in_channels=n_feat * t, out_channels=n_feat, kernel_size=1, padding=0, groups=1)
        self.res_scale = res_scale

    def forward(self, x):
        
        pc1 = self.point_conv1(x)
        a1 = self.relu(pc1)
        dc1 = self.depth_conv(a1)
        a2 = self.relu(dc1)
        pc2 = self.point_conv2(a2)

        res = self.ca(pc2)
        res += x
        return res
        
## MnasNet
class MNASNET(nn.Module):
    def __init__(self,n_feat,reduction=16,act=nn.ReLU(True), res_scale=1,t=2):
        super(MNASNET, self).__init__()
        self.ca = CALayer(n_feat*t, reduction)
        self.relu = act
        self.depth_conv = nn.Conv2d(in_channels=n_feat*t,out_channels=n_feat*t,kernel_size=3,stride=1,padding=1,groups=n_feat*t)
        self.point_conv1 = nn.Conv2d(in_channels=n_feat,out_channels=n_feat*t,kernel_size=1,padding=0,groups=1)
        self.point_conv2 = nn.Conv2d(in_channels=n_feat * t, out_channels=n_feat, kernel_size=1, padding=0, groups=1)
        self.res_scale = res_scale

    def forward(self, x):

        pc1 = self.point_conv1(x)
        a1 = self.relu(pc1)
        dc1 = self.depth_conv(a1)
        a2 = self.relu(dc1)

        ca = self.ca(a2)
        res = self.point_conv2(ca)

        # res = self.ca(pc2)
        res += x
        return res

## Residual Channel Attention Block (ECAB)
class ECAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ECAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(EcaLayer(channels=n_feat))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class EcaLayer(nn.Module):

    def __init__(self, channels, gamma=2, b=1):
        super(EcaLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)
        
class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=4):

        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   # 
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  # 
        for i in range(M):
            # 
            self.conv.append(nn.Sequential(nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=4,bias=False),
                                           # nn.BatchNorm2d(out_channels),
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               # nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)  # 
        self.softmax=nn.Softmax(dim=1) # 
    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output) # 
        s=self.global_pool(U)
        z=self.fc1(s)  # S->Z
        a_b=self.fc2(z) # Z->a£¬b 
        #a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1) #
        a_b = a_b.view(batch_size, self.M, self.out_channels, -1)
        
        a_b=self.softmax(a_b) # 
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        #a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) # 
        a_b = list(map(lambda x: x.view(batch_size, self.out_channels, 1, 1), a_b))
        V=list(map(lambda x,y:x*y,output,a_b)) # 
        V=reduce(lambda x,y:x+y,V) # 
        return V


## Residual Selective Kernel Block (RSKB)
class RSKB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size,bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RSKB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(SKConv(in_channels=n_feat,out_channels=n_feat))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class DepthWiseConv(nn.Module):
    def __init__(self, n_feat):
        super(DepthWiseConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, stride=1, padding=1,groups=n_feat)
        self.point_conv = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=1, stride=1,padding=0, groups=1)

    def forward(self, x):
        y = self.depth_conv(x)
        y = self.point_conv(y)
        return y

## DWRCAB
class DWRCAB(nn.Module):
    def __init__(self,n_feat,reduction=16,bn=False, act=nn.ReLU(True), res_scale=1):
        super(DWRCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(DepthWiseConv(n_feat))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## DWRCAB2
class DWRCAB2(nn.Module):
    def __init__(self,n_feat,reduction=16,bn=False, act=nn.ReLU(True), res_scale=1):
        super(DWRCAB2, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(DepthWiseConv(n_feat))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            modules_body.append(act)
        modules_body.append(default_conv(n_feat,n_feat,1))
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class PAConv(nn.Module):

    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1)  # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)  # 3x3 convolution

    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)

        out = torch.mul(self.k3(x), y)
        out = self.k4(out)

        return out


class SCPA(nn.Module):
    """SCPA is modified from SCNet (Jiang-Jiang Liu et al. Improving Convolutional Networks with Self-Calibrated Convolutions. In CVPR, 2020)
        Github: https://github.com/MCG-NKU/SCNet
    """

    def __init__(self, nf, reduction=2, stride=1, dilation=1):
        super(SCPA, self).__init__()
        group_width = nf // reduction

        self.conv1_a = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)
        self.conv1_b = nn.Conv2d(nf, group_width, kernel_size=1, bias=False)

        self.k1 = nn.Sequential(
            nn.Conv2d(
                group_width, group_width, kernel_size=3, stride=stride,
                padding=dilation, dilation=dilation,
                bias=False)
        )

        self.PAConv = PAConv(group_width)

        self.conv3 = nn.Conv2d(
            group_width * reduction, nf, kernel_size=1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_b = self.conv1_b(x)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out_a = self.k1(out_a)
        out_b = self.PAConv(out_b)
        out_a = self.lrelu(out_a)
        out_b = self.lrelu(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))
        out += residual

        return out
        
## Residual Channel Attention Block (RCAB)
class RCAB_G(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_G, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=4,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
        
class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        
        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)
        
## RCAB_G_CS: Group Conv with Channel shuffle
class RCAB_G_CS(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_G_CS, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=4,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(ShuffleBlock(groups=4))
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class RCAB_G2(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_G2, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=6,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(act)
        modules_body.append(default_conv(n_feat, n_feat, 1))
        
        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class RCAB_G2_CS(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB_G2_CS, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=4,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(act)
        modules_body.append(ShuffleBlock(groups=4))
        modules_body.append(default_conv(n_feat, n_feat, 1))

        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res
        
class RB_G3(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RB_G3, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=4,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(act)
        modules_body.append(ShuffleBlock(groups=4))
        # modules_body.append(default_conv(n_feat, n_feat, 1))

        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RB_G4(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RB_G4, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=1,groups=4,bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(act)
        # modules_body.append(ShuffleBlock(groups=4))
        modules_body.append(default_conv(n_feat, n_feat, 1))

        modules_body.append(CALayer(n_feat, reduction))

        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res