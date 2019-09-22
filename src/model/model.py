import os
import sys
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('model/',''))
import torch.nn as nn
import numpy as np
import torch
from utils import util
from utils.SMPL import SMPL
from config import args

from torch.nn import functional as F

import math
import cv2
import tqdm
import json
import time
import utils.neuralrenderer_render as nr
import random
import math
from torch.nn.utils import weight_norm

BN_MOMENTUM = 0.1

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = 'module.',not_print=False):
    success_layer = []
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                if not not_print:
                    print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
            if prefix in k and prefix!='':
                k=k.split(prefix)[1]
            success_layer.append(k)
        except:
            if not not_print:
                print('copy param {} failed'.format(k))
            continue
    return success_layer

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)#,affine=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)#,affine=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                momentum=BN_MOMENTUM)#,affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class PoseResNet(nn.Module):

    def __init__(self, block, layers, baseline0=False, baseline=False,baseline2=False,kp3d=False,video_clips_input=False,**kwargs):
        self.inplanes = 64
        super(PoseResNet, self).__init__()
        self.baseline0=baseline0
        self.baseline = baseline
        self.baseline2 = baseline2
        self.video_clips_input = video_clips_input
        self.kp3d = kp3d
        if self.video_clips_input:
            self.conv1 = nn.Sequential(
                nn.Conv3d(3, 64, kernel_size=3, stride=(1,2,2), padding=(0,1,1), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(inplace=True),
                nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=(0,1,1), bias=False))
            nn.init.kaiming_normal_(self.conv1[0].weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.conv1[3].weight, mode='fan_out', nonlinearity='relu')
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)#,affine=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if not kp3d:
            self.deconv_layers = self._make_deconv_layer(3,(256,256,256),(4,4,4), )
            self.final_layer = nn.Conv2d(
                in_channels=256,out_channels=14,kernel_size=1,
                stride=1,padding=0)
        else:
            self.deconv_head_kp3d = DeconvHead(2048, 3, 256, 4, 1, 14, 14)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.smpl = SMPL(args.smpl_model, joint_type = 'lsp',obj_saveable = True)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),)#,affine=False),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            if i==0:
                self.inplanes=2048
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))#,affine=False))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x, kps_gt=None, details=None,allfeature=False,blf=None,only_blf=False,kp3d_24=False):
        x = self.conv1(x)
        if self.video_clips_input:
            x = x.squeeze()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if details is None:
            x_details = self.avgpool(x)
            x_details = x_details.view(x_details.size(0), -1)
            details = self.fc1(x_details)

        x_kp = self.deconv_layers(x)
        x_kp = self.final_layer(x_kp)
        kps = softmax_integral_tensor(x_kp)
        kps_gt = torch.where(kps_gt!=-2.,kps_gt, kps) if kps_gt is not None else kps

        x = self.bilinear(kps_gt,details)
        params = self.fc(x)
        if only_blf:
            return x,kps,params

        out = self._calc_detail_info(params,kp3d_24=kp3d_24)
        if allfeature:
            return out,kps,details,x_details,x,params
        return out,kps,details

    def _calc_detail_info(self, param,kp3d_24=False):

        cam = param[:, 0:3].contiguous()
        pose = param[:, 3:75].contiguous()
        shape = param[:, 75:].contiguous()
        verts, j3d, Rs = self.smpl(beta = shape, param = pose, get_skin = True)
        projected_j2d = util.batch_orth_proj(j3d.clone(), cam, mode='2d')
        j3d = util.batch_orth_proj(j3d.clone(), cam, mode='j3d')
        verts_camed = util.batch_orth_proj(verts, cam, mode='v3d')
        if kp3d_24:
            _, j3d, _ = self.smpl(beta = shape, param = pose, get_org_joints = True)
            j3d = batch_orth_proj(j3d.clone(), cam, mode='3d')

        return ((cam,pose,shape), verts, projected_j2d, j3d, Rs,verts_camed,j3d)

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

def get_pose_net(num_layers = 50,baseline0=False, baseline=False,baseline2=False,kp3d=False,video_clips_input=False):

    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, baseline0=baseline0, baseline=baseline, baseline2=baseline2,kp3d=kp3d,video_clips_input=video_clips_input)

    '''
    if num_layers==50:
        pretrained_path = "/export/home/suny/home/pose_tracking/models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar"
        prefix = ''
    
        pretrained_state_dict = torch.load(pretrained_path)
        print('using pre-trained modle from: ', pretrained_path)

        success_layer = copy_state_dict(model.state_dict(), pretrained_state_dict, prefix = prefix,not_print=True)#module.
        for layer_name in success_layer:
            if 'layer' in layer_name and len(layer_name.split('.'))>2:
                if len(layer_name.split('.'))==3:
                    eval('model.{}[{}].{}'.format(*layer_name.split('.'))).requires_grad = False
                elif len(layer_name.split('.'))==4:
                    eval('model.{}[{}].{}.{}'.format(*layer_name.split('.'))).requires_grad = False
                elif len(layer_name.split('.'))==5:
                    eval('model.{}[{}].{}[{}].{}'.format(*layer_name.split('.'))).requires_grad = False
                else:
                    print('Error layer name split larger than 5')
                    print(layer_name)
            elif 'deconv_head_kp3d' in layer_name:
                if len(layer_name.split('.'))==4:
                    eval('model.{}.{}[{}].{}'.format(*layer_name.split('.'))).requires_grad = False
                else:
                    print('Error layer name split larger than 4')
                    print(layer_name)
            else:
                eval('model.'+layer_name).requires_grad = False
    else:
        print('Only surpport the resnet-50 as pretrain model!')
    '''

    model.fc1 = nn.Linear(2048, 512)
    nn.init.kaiming_normal_(model.fc1.weight, mode='fan_out', nonlinearity='relu')
    model.bilinear = nn.Bilinear(28,512,512,bias=False)
    nn.init.kaiming_normal_(model.bilinear.weight, mode='fan_out', nonlinearity='relu')
    model.fc = nn.Linear(512, 85)

    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    print('Pretrain loaded!')

    return model


def generate_2d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_y = heatmaps.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).float(), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).float(), devices=[accu_y.device.index])[0]
    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)

    return accu_x, accu_y


def softmax_integral_tensor(preds, num_joints=14, hm_width=64, hm_height=64):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # integrate heatmap into joint location
    x, y = generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    x = 2* x / float(hm_width) - 1
    y = 2* y / float(hm_height) - 1

    preds = torch.cat((x, y), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 2))
    return preds

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)

    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))

    accu_x = heatmaps.sum(dim=2)
    accu_x = accu_x.sum(dim=2)
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)

    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).float(), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).float(), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim).float(), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)

    return accu_x, accu_y, accu_z


def softmax_integral_tensor_3d(preds, num_joints=14, hm_width=64, hm_height=64, hm_depth=14):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1))
    preds = F.softmax(preds, 2)

    # integrate heatmap into joint location
    x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    x = 2*x / float(hm_width) - 1
    y = 2*y / float(hm_height) - 1
    z = 2*z / float(hm_depth) - 1
    preds = torch.cat((x, y, z), dim=2)
    preds = preds.reshape((preds.shape[0], num_joints * 3))
    return preds

class DeconvHead(nn.Module):
    def __init__(self, in_channels, num_layers, num_filters, kernel_size, conv_kernel_size, num_joints, depth_dim,
                 with_bias_end=True):
        super(DeconvHead, self).__init__()

        conv_num_filters = num_joints * depth_dim

        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, 'Only support kenerl 2, 3 and 4'
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert conv_kernel_size == 1 or conv_kernel_size == 3, 'Only support kenerl 1 and 3'
        if conv_kernel_size == 1:
            pad = 0
        elif conv_kernel_size == 3:
            pad = 1

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                                   output_padding=output_padding, bias=False))
            self.features.append(nn.BatchNorm2d(num_filters))#,affine=False))
            self.features.append(nn.ReLU(inplace=True))

        if with_bias_end:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=True))
        else:
            self.features.append(
                nn.Conv2d(num_filters, conv_num_filters, kernel_size=conv_kernel_size, padding=pad, bias=False))
            self.features.append(nn.BatchNorm2d(conv_num_filter))#,affine=False))
            self.features.append(nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if with_bias_end:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)

    def forward(self, x):
        for i, l in enumerate(self.features):
            x = l(x)
        return x

if __name__ == '__main__':
    net = get_pose_net(video_clips_input=False).cuda()
    nx = torch.rand(4, 3, 256, 256).float().cuda()
    kps = torch.rand(4, 28).float().cuda()
    kps[:,2:8]=-2
    y = net(nx,kps_gt=kps)
    print(y[0][1].shape)