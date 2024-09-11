import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from .common import MeanShift, std, default_conv, CEALayer, default_conv_stride2
# from IPython import embed

def make_model(args, parent=False):
    return MODEL(args)

class Scale(nn.Module):

    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU()
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer
    
    
class SRBW1(nn.Module):
    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True)):
        super(SRBW1, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            nn.Conv2d(n_feats , n_feats*2 , kernel_size=1, padding=0))
        body.append(act)
        body.append(
            nn.Conv2d(n_feats*2, n_feats//2, kernel_size=1, padding=0))
        body.append(
            nn.Conv2d(n_feats//2, n_feats, kernel_size=3, padding=1))

        self.body = nn.Sequential(*body)
        self.SAlayer = sa_layer(n_feats)

    def forward(self, x):
        y = self.res_scale(self.SAlayer(self.body(x))) + self.x_scale(x)
        return y
        

class SRBW2(nn.Module):
    def __init__(
            self, n_feats, wn=lambda x: torch.nn.utils.weight_norm(x), act=nn.ReLU(True)):
        super(SRBW2, self).__init__()
        self.res_scale = Scale(1)
        self.x_scale = Scale(1)
        body = []
        body.append(
            nn.Conv2d(n_feats , n_feats*2 , kernel_size=1, padding=0))
        body.append(act)
        body.append(
            nn.Conv2d(n_feats*2, n_feats//2, kernel_size=1, padding=0))
        body.append(
            nn.Conv2d(n_feats//2, n_feats//2, kernel_size=3, padding=1))

        self.body = nn.Sequential(*body)
        self.SAlayer = sa_layer(n_feats//2)
        self.conv = nn.Conv2d(n_feats , n_feats//2 , kernel_size=3, padding=1)

    def forward(self, x):
        y = self.res_scale(self.SAlayer(self.body(x))) + self.x_scale(self.conv(x))
        return y
        

class CoffConv(nn.Module):
    def __init__(self, n_feats):
        super(CoffConv, self).__init__()
        self.upper_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(n_feats, n_feats // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // 8, n_feats, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        self.std = std
        self.lower_branch = nn.Sequential(
            nn.Conv2d(n_feats, n_feats // 8, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats // 8, n_feats, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, fea):
        upper = self.upper_branch(fea)
        lower = self.std(fea)
        lower = self.lower_branch(lower)

        out = torch.add(upper, lower) / 2

        return out


class sa_layer(nn.Module):
    def __init__(self, n_feats, groups=4):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, n_feats // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, n_feats // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(n_feats // (2 * groups), n_feats // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out


class MY(nn.Module):
    def __init__(self, n_feats, act=nn.ReLU(True)):
        super(MY, self).__init__()

        self.act = activation('lrelu', neg_slope=0.05)
        wn = lambda x: torch.nn.utils.weight_norm(x)
        self.srb1 = SRBW1(n_feats)
        self.srb2 = SRBW1(n_feats)
        self.rb1 = SRBW1(n_feats)
        self.rb2 = SRBW1(n_feats)
        self.A1_coffconv = CoffConv(n_feats)
        self.B1_coffconv = CoffConv(n_feats)
        self.A2_coffconv = CoffConv(n_feats)
        self.B2_coffconv = CoffConv(n_feats)
        self.conv_distilled1 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_distilled2 = nn.Conv2d(n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.sigmoid1 = nn.Sigmoid()
        self.sigmoid2 = nn.Sigmoid()
        self.sigmoid3 = nn.Sigmoid()
        self.scale_x1 = Scale(1)
        self.scale_x2 = Scale(1)
        self.srb3 = SRBW1(n_feats)
        self.srb4 = SRBW1(n_feats)
        self.fuse1 = SRBW2(n_feats*2)
        self.fuse2 = nn.Conv2d(2*n_feats, n_feats, kernel_size=1, stride=1, padding=0, bias=False, dilation=1)

    def forward(self, x):
        out_a = self.act(self.srb1(x))
        distilled_a1 = remaining_a1 = out_a
        out_a = self.rb1(remaining_a1) 
        A1 = self.A1_coffconv(out_a)
        out_b_1 = A1 * out_a + x
        B1 = self.B1_coffconv(x)
        out_a_1 = B1 * x + out_a

        out_b = self.act(self.srb2(out_b_1))
        distilled_b1 = remaining_b1 = out_b
        out_b = self.rb2(remaining_b1)
        A2 = self.A2_coffconv(out_a_1)
        out_b_2 = A2 * out_a_1 + out_b
        out_b_2 = out_b_2 * self.sigmoid1(self.conv_distilled1(distilled_b1))
        B2 = self.B2_coffconv(out_b)
        out_a_2 = out_b * B2 + out_a_1
        out_a_2 = out_a_2 * self.sigmoid2(self.conv_distilled2(distilled_a1))

        out_a_out = self.srb3(out_a_2)
        out_b_out = self.srb4(out_b_2)

        out1 = self.fuse1(torch.cat([self.scale_x1(out_a_out), self.scale_x2(out_b_out)], dim=1))
        out2 = self.sigmoid3(self.fuse2(torch.cat([self.scale_x1(out_a_out), self.scale_x2(out_b_out)], dim=1)))
        
        out = out2 * out_b_out
        y1 = out1 + out

        return y1


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]  ##rows
    out_cols = (cols + strides[1] - 1) // strides[1]  ##cols
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1  ##3
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1  ##3
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)  ##2
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)  ##2
    # Pad the input
    padding_top = int(padding_rows / 2.)  ##1
    padding_left = int(padding_cols / 2.) ##1
    padding_bottom = padding_rows - padding_top  ##1
    padding_right = padding_cols - padding_left  ##1
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    batch_size, channel, height, width = images.size()
    
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             dilation=rates,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks
    
    
def reverse_patches(images, out_size, ksizes, strides, padding):
    unfold = torch.nn.Fold(output_size = out_size, 
                            kernel_size=ksizes, 
                            dilation=1, 
                            padding=padding, 
                            stride=strides)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features//4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups

        return x.view(N,g,int(C/g),H,W).permute(0,2,1,3,4).contiguous().view(N,C,H,W)


class EffAttention(nn.Module):
    def __init__(self, dim, num_heads=9, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  ##288//8 == 36
        self.scale = qk_scale or head_dim ** -0.5  ##1/6

        self.reduce = nn.Linear(dim, dim//2, bias=qkv_bias)
        self.qkv = nn.Linear(dim//2, dim//2 * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim//2, dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        x = self.reduce(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  ##3, B, 8, N, C//8
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q_all = torch.split(q, math.ceil(N//4), dim=-2)    #1, B, 8, N//4, 4, C//8
        k_all = torch.split(k, math.ceil(N//4), dim=-2)
        v_all = torch.split(v, math.ceil(N//4), dim=-2)    

        output = []
        for q,k,v in zip(q_all, k_all, v_all):
            attn = (q @ k.transpose(-2, -1)) * self.scale   #16*8*37*37
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            trans_x = (attn @ v).transpose(1, 2)
            output.append(trans_x)
        x = torch.cat(output,dim=1)
        #x = x.view(B,math.ceil(N//math.ceil(N//4)),math.ceil(N//4),self.num_heads,C // self.num_heads).permute(0,2,1,3,4).contiguous().view(B,N,self.num_heads,C // self.num_heads)
        x = x.reshape(B,N,C)
        x = self.proj(x)
        return x


class TransBlock(nn.Module):
    def __init__(
        self, n_feat = 64,dim=768, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super(TransBlock, self).__init__()
        self.dim = dim
        self.atten = EffAttention(self.dim, num_heads=9, qkv_bias=False, qk_scale=None,
                             attn_drop=0., proj_drop=0.)
        self.norm1 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim//4, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x):
        b,c,h,w = x.shape
        x = extract_image_patches(x, ksizes=[3, 3],
                                      strides=[1,1],
                                      rates=[1, 1],
                                      padding='same')
        x = x.permute(0,2,1)
        x = x + self.atten(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        x = x.permute(0,2,1)
        x = reverse_patches(x, (h,w), (3,3), 1, 1)
        
        return x


class MODEL(nn.Module):
    def __init__(self,args,conv=default_conv):
        super(MODEL, self).__init__()
        n_feats = 32
        kernel_size = 3
        n_colors = 3
        out_channels = 3
        upscale = 4
        self.act = activation('lrelu', neg_slope=0.05)
        self.Upsample = nn.Upsample(scale_factor=4,
                                    mode='bilinear', align_corners=False)
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.Sub_mean = MeanShift(255, rgb_mean, rgb_std)
        self.Add_mean = MeanShift(255, rgb_mean, rgb_std, 1)
        self.head = conv(n_colors, n_feats , kernel_size) 
        
        self.MY1 = MY(n_feats )
        self.MY2 = MY(n_feats )
        self.MY3 = MY(n_feats )
        self.conv_concat1 = nn.Conv2d(2*n_feats , n_feats , kernel_size=3, stride=1, padding=1, groups=4, bias=False, dilation=1)
        self.conv_concat2 = nn.Conv2d(2*n_feats , n_feats , kernel_size=3, stride=1, padding=1, groups=4, bias=False, dilation=1)
        self.channel_shuffle1 = ShuffleBlock(groups=4)
        self.channel_shuffle2 = ShuffleBlock(groups=4)
        
        self.attention1 = TransBlock(n_feat=n_feats, dim=n_feats*9)
        self.attention2 = TransBlock(n_feat=n_feats, dim=n_feats*9)
        self.conv = nn.Conv2d(n_feats*2, n_feats, kernel_size=1, stride=1, padding=0, bias=False) 
        
        self.scale_x = Scale(0.5)
        self.scale_res = Scale(0.5)
        self.conv_down = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False) 
        self.conv_up = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False) 
        
        self.conv1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=False)
        up_body = []
        up_body.append(default_conv(n_feats, out_channels*upscale*upscale, kernel_size=3, bias=True))
        up_body.append(nn.PixelShuffle(upscale))
        self.UP1 = nn.Sequential(*up_body)
        
        self.conv2 = conv(n_colors, n_feats , kernel_size)
        up_body = []
        up_body.append(default_conv(n_feats,out_channels*upscale*upscale, kernel_size=3, bias=True))
        up_body.append(nn.PixelShuffle(upscale))
        self.UP2 = nn.Sequential(*up_body)
        
    def forward(self, x):
        y_input0 = self.Sub_mean(x)
        y_input = self.head(y_input0) 
        
        res = y_input
        out1 = self.MY1(y_input)
        out2 = self.MY2(out1)
        out3 = self.MY3(out2)
        
        out_concat1 = self.channel_shuffle1(self.conv_concat1(torch.cat([out1, out2], dim=1)))
        out_concat2 = self.channel_shuffle2(self.conv_concat2(torch.cat([out_concat1, out3], dim=1)))                     
        x = self.scale_x(out_concat2 + out3) + self.scale_res(res)
        
        out2_trans = self.attention1(self.conv_down(out2) + y_input)
        out2_trans = self.attention2(out2_trans)
        
        res = x
        out_1 = self.MY1(x) # self.MY1() share parameters with out1
        out_2 = self.MY2(out_1 + self.conv_up(out2_trans)) # self.MY2() share parameters with out2
        out_3 = self.MY3(out_2) # self.MY3() share parameters with out3
        
        out_concat_1 = self.channel_shuffle1(self.conv_concat1(torch.cat([out_1, out_2], dim=1)))
        out_concat_2 = self.channel_shuffle2(self.conv_concat2(torch.cat([out_concat_1, out_3], dim=1)))
        out = self.scale_x(out_concat_2 + out_3) + self.scale_res(res)
        
        y_final = self.conv(torch.cat([out, out2_trans], dim=1))
        y1 = self.UP1(self.conv1(y_final))
        y2 = self.UP2(self.conv2(y_input0))
        y = y1 + y2
        output = self.Add_mean(y)

        return output
