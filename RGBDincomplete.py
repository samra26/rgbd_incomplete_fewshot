import torch
import torch.nn as nn
from cswin import CSWinTransformer
import torch.nn.functional as F
from functools import partial
from torchsummary import summary
from timm.models.layers import DropPath, trunc_normal_
import os
import cv2
import numpy
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
im_size=(320,320)


class FCU(nn.Module):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes,  act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCU, self).__init__()

        
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x):
        x_r = self.act(self.bn(self.conv_project(x)))

        return x_r

class RGBDInModule(nn.Module):
    def __init__(self, backbone,embed_dim):
        super(RGBDInModule, self).__init__()
        self.backbone = backbone
        self.dim = embed_dim
        for i in range(4):
            self.add_module('expand_block_' + str(i), FCU(self.dim, self.dim))

        

    def load_pretrained_model(self, model_path):
        pretrained_dict = torch.load(model_path)
        model_dict = self.backbone.state_dict()
        pretrained_dict = {ka: va for ka, va in pretrained_dict.items() if ka in model_dict}
        model_dict.update(pretrained_dict)
        self.backbone.load_state_dict(model_dict, strict=True)
        

    def forward(self, x):
        feature_stage=[]
        x,x1= self.backbone(x)
        for i in range(len(x1)):
            print("stage",i,x1[i].shape)
        a=[1,5,37,40]
        count=0
        for i in a:
            #print(i,'The backbone features are',x1[i].shape)
            x_r=eval('self.expand_block_' + str(count))(x1[i])
            #print(i,x_r.shape)
            count=count+1
            feature_stage.append(x_r)
            

        return feature_stage
        


class RGBD_incomplete(nn.Module):
    def __init__(self,embed_dim,RGBDInModule):
        super(RGBD_incomplete, self).__init__()
        
        self.RGBDInModule = RGBDInModule
        self.embed_dim=embed_dim
        self.relu = nn.ReLU(inplace=True)
        self.conv_stage1=nn.Sequential(nn.Conv2d(self.embed_dim, self.embed_dim, 1), self.relu)
        self.conv_stage2=nn.Sequential(nn.Conv2d(self.embed_dim*2, self.embed_dim, 1), self.relu)
        self.conv_stage3=nn.Sequential(nn.Conv2d(self.embed_dim*4, self.embed_dim, 1), self.relu)
        self.conv_stage4=nn.Sequential(nn.Conv2d(self.embed_dim*8, self.embed_dim, 1), self.relu)
        self.deconv_stage1=nn.ConvTranspose2d(self.embed_dim,1,kernel_size=3, stride=4, padding=0, output_padding=1, dilation=1)
        self.deconv_stage2=nn.ConvTranspose2d(self.embed_dim,1,kernel_size=3, stride=8, padding=0, output_padding=3, dilation=2)
        self.deconv_stage3=nn.ConvTranspose2d(self.embed_dim,1,kernel_size=5, stride=16, padding=0, output_padding=3, dilation=3)
        self.deconv_stage4=nn.ConvTranspose2d(self.embed_dim,1,kernel_size=7, stride=32, padding=1, output_padding=3, dilation=5)
        self.last_conv=nn.Conv2d(4,1,1,1)

        
    def forward(self, f_all):
        feat_rgb = self.RGBDInModule(f_all)
        
        rgb_branch1 = self.conv_stage1(feat_rgb[0])
        rgb_branch2 = self.conv_stage2(feat_rgb[1])
        rgb_branch3 = self.conv_stage3(feat_rgb[2])
        rgb_branch4 = self.conv_stage4(feat_rgb[3])
        rgb_out1 = self.deconv_stage1(rgb_branch1)
        rgb_out2 = self.deconv_stage2(rgb_branch2)
        rgb_out3 = self.deconv_stage3(rgb_branch3)
        rgb_out4 = self.deconv_stage4(rgb_branch4)
        print(rgb_branch1.shape,rgb_out1.shape)
        print(rgb_branch2.shape,rgb_out2.shape)
        print(rgb_branch3.shape,rgb_out3.shape)
        print(rgb_branch4.shape,rgb_out4.shape)
        
        feat_rgb_out=self.last_conv(torch.cat((rgb_out1,rgb_out2,rgb_out3,rgb_out4),dim=1))
        print(feat_rgb_out.shape)
        
        return feat_rgb_out
        


def build_model(network, base_model_cfg,embed_dim):
    backbone = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[4,8,16,32], mlp_ratio=4.0)
      
   

    return RGBD_incomplete(embed_dim,RGBDInModule(backbone,embed_dim))
