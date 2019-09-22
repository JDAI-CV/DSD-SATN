import torch.nn as nn
import torch
import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('model/',''))
from utils.SMPL import SMPL
from utils import util
from config import args
from transformer.Models import *
import torch.nn.functional as F

class Transform_Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, batch_size=16, spwan=27, d_word_vec=512,
            n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1):

        super().__init__()
        self.spwan =spwan

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.slf_attn_mask = torch.zeros((batch_size,self.spwan,self.spwan)).byte().cuda()
        self.non_pad_mask = torch.ones((batch_size,self.spwan,1)).float().cuda()
        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(spwan+1, d_word_vec, padding_idx=0),freeze=True).cuda()
        self.position_encoding = self.position_enc(torch.arange(spwan).cuda())

        print('Created Transform Encoder')

    def forward(self, enc_output, return_attns=False):
        enc_slf_attn_list = []

        if self.slf_attn_mask.shape[0]!=enc_output.shape[0]:
            self.slf_attn_mask = torch.zeros((enc_output.shape[0],self.spwan,self.spwan)).byte().cuda()
            self.non_pad_mask = torch.ones((enc_output.shape[0],self.spwan,1)).float().cuda()
        enc_output = enc_output + self.position_encoding

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=self.non_pad_mask,
                slf_attn_mask=self.slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class TemporalModelBase(nn.Module):
    """
    Do not instantiate this class.
    """

    def __init__(self,in_features,filter_widths, causal, dropout, channels,with_transformer_encoder=False,fc_summon=False,fc_dim=512):
        super().__init__()
        self.with_transformer_encoder = with_transformer_encoder
        self.fc_summon=fc_summon
        if self.with_transformer_encoder:
            self.transformer_encoder = Transform_Encoder(d_model=in_features,batch_size=16, spwan=pow(3,len(filter_widths)),n_layers=2, )

        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.in_features = in_features
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        if self.fc_summon:
            self.shrink = nn.Sequential(
                nn.Linear(channels+fc_dim, 2048),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(2048, 2048),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(2048, 85),)
        else:
            self.shrink = nn.Conv1d(channels, 85, 1)

        self.smpl = SMPL(args.smpl_model, joint_type = 'lsp',obj_saveable = True)

    def set_bn_momentum(self, momentum):
        self.expand_bn.momentum = momentum
        for bn in self.layers_bn:
            bn.momentum = momentum

    def receptive_field(self):
        """
        Return the total receptive field of this model as # of frames.
        """
        frames = 0
        for f in self.pad:
            frames += f
        return 1 + 2*frames

    def total_causal_shift(self):
        """
        Return the asymmetric offset for sequence padding.
        The returned value is typically 0 if causal convolutions are disabled,
        otherwise it is half the receptive field.
        """
        frames = self.causal_shift[0]
        next_dilation = self.filter_widths[0]
        for i in range(1, len(self.filter_widths)):
            frames += self.causal_shift[i] * next_dilation
            next_dilation *= self.filter_widths[i]
        return frames

    def forward(self, x, just_out=False,kp3d_24=False, fcc=None):
        if self.with_transformer_encoder:
            x=self.transformer_encoder(x)
        x = x.permute(0, 2, 1)

        if self.fc_summon:
            x = self._forward_blocks(x, fcc=fcc)
        else:
            x = self._forward_blocks(x)
            x = x.squeeze()
        out = self._calc_detail_info(x,kp3d_24=kp3d_24)
        if just_out:
            return out

        return out,out[2],None

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

class TemporalModel(TemporalModelBase):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, in_features, filter_widths=[3,3], causal=False, dropout=0.25, channels=1024, dense=False,with_transformer_encoder=False,deformable=False,bigger_view=True,fc_summon=False,fc_dim=512):
        """
        Initialize this model.

        Arguments:
        in_features -- number of input features
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__(in_features, filter_widths, causal, dropout, channels,with_transformer_encoder,fc_summon,fc_dim)
        self.deformable = deformable
        self.fc_summon = fc_summon

        if self.deformable:
            self.expand_conv = DeformConv1d(in_features, channels, kernel_size=filter_widths[0], bias=False,bigger_view=bigger_view)
        else:
            self.expand_conv = nn.Conv1d(in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)

            if self.deformable:
                layers_conv.append(DeformConv1d(channels, channels,
                                         kernel_size=filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False,bigger_view=bigger_view))
            else:
                layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            if i !=len(filter_widths)-1:
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            if i !=len(filter_widths)-1:
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

        print('Created Temporal Model')

    def _forward_blocks(self, x, fcc=None):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.layers_conv[2*i](x)
            if i!=len(self.pad) - 2:
                x = self.drop(self.relu(self.layers_bn[2*i](x)))

            x = self.layers_conv[2*i + 1](x)
            if i!=len(self.pad) - 2:
                x = self.drop(self.relu(self.layers_bn[2*i](x)))
            x = res + x

        if fcc is not None:
            x = x.squeeze()
            x = torch.cat([x,fcc],dim=-1)
        x = self.shrink(x)
        return x


class Time_warping_FPN(nn.Module):
    """
    Reference 3D pose estimation model with temporal convolutions.
    This implementation can be used for all use-cases.
    """

    def __init__(self, spawn=27, in_features=512, filter_widths=[3,3,3], causal=False, dropout=0.25, channels=512, dense=False,with_transformer_encoder=False,deformable=False,bigger_view=False):
        """
        Initialize this model.

        Arguments:
        in_features -- number of input features
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        dense -- use regular dense convolutions instead of dilated convolutions (ablation experiment)
        """
        super().__init__()
        self.deformable = deformable
        self.in_features = in_features
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [ filter_widths[0] // 2 ]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)

        if self.deformable:
            self.expand_conv = DeformConv1d(in_features, channels, kernel_size=filter_widths[0], bias=False,bigger_view=bigger_view)
        else:
            self.expand_conv = nn.Conv1d(in_features, channels, filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [ (filter_widths[0]) // 2 if causal else 0 ]
        next_dilation = 1
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2 * next_dilation) if causal else 0)

            if self.deformable:
                layers_conv.append(DeformConv1d(channels, channels,
                                         kernel_size=filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False,bigger_view=bigger_view))
            else:
                layers_conv.append(nn.Conv1d(channels, channels,
                                         filter_widths[i] if not dense else (2*self.pad[-1] + 1),
                                         dilation=next_dilation if not dense else 1,
                                         bias=False))
            if i !=len(filter_widths)-1:
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            if i !=len(filter_widths)-1:
                layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

        self.subnet1_1 = nn.Sequential(
            nn.Conv1d(channels,channels,3,bias=False,padding=1),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False),)

        self.subnet1_2 = nn.Sequential(
            nn.Conv1d(channels,channels,3,bias=False,padding=1),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False))

        self.subnet2_1 = nn.Sequential(
            nn.Conv1d(channels,channels,3,bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False),)

        self.subnet2_2 = nn.Sequential(
            nn.Conv1d(channels,channels,3,bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False))

        self.subnet3_1 = nn.Sequential(
            nn.Conv1d(channels,channels,5,bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False),)

        self.subnet3_2 = nn.Sequential(
            nn.Conv1d(channels,channels,5,bias=False),
            nn.BatchNorm1d(channels, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Conv1d(channels,channels,1,bias=False))

        print('Created Temporal Model')

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))
        fpn_cache = [x]

        for i in range(len(self.pad) - 1):
            pad = self.pad[i+1]
            shift = self.causal_shift[i+1]
            res = x[:, :, pad + shift : x.shape[2] - pad + shift]

            x = self.layers_conv[2*i](x)
            if i!=len(self.pad) - 2:
                x = self.drop(self.relu(self.layers_bn[2*i](x)))

            x = self.layers_conv[2*i + 1](x)
            if i!=len(self.pad) - 2:
                x = self.drop(self.relu(self.layers_bn[2*i](x)))
            x = res + x

            fpn_cache.append(x)

        feature_pyramid = fpn_cache[-1]
        long_term_1 = (F.softmax(self.subnet1_1(feature_pyramid))*self.subnet1_2(feature_pyramid)).sum(-1)

        feature_pyramid = F.upsample(feature_pyramid,size=5) + fpn_cache[-2]
        long_term_2 = (F.softmax(self.subnet2_1(feature_pyramid))*self.subnet2_2(feature_pyramid)).sum(-1)

        feature_pyramid = F.upsample(feature_pyramid,size=7) + fpn_cache[-3]
        long_term_3 = (F.softmax(self.subnet3_1(feature_pyramid))*self.subnet3_2(feature_pyramid)).sum(-1)

        fp = torch.stack([long_term_1,long_term_2,long_term_3],dim=-1)
        fp = fp.permute(0, 2, 1)

        return fp


class TemporalModelOptimized1f(TemporalModelBase):
    """
    3D pose estimation model optimized for single-frame batching, i.e.
    where batches have input length = receptive field, and output length = 1.
    This scenario is only used for training when stride == 1.
    This implementation replaces dilated convolutions with strided convolutions
    to avoid generating unused intermediate results. The weights are interchangeable
    with the reference implementation.
    """

    def __init__(self, in_features, filter_widths, causal=False, dropout=0.25, channels=1024):
        """
        Initialize this model.
        Arguments:
        in_features -- number of input features
        filter_widths -- list of convolution widths, which also determines the # of blocks and receptive field
        causal -- use causal convolutions instead of symmetric convolutions (for real-time applications)
        dropout -- dropout probability
        channels -- number of convolution channels
        """
        super().__init__(in_features, filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(in_features, channels, filter_widths[0], stride=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [ (filter_widths[0] // 2) if causal else 0 ]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1)*next_dilation // 2)
            self.causal_shift.append((filter_widths[i]//2) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i], stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i+1] + self.filter_widths[i+1]//2 :: self.filter_widths[i+1]]

            x = self.drop(self.relu(self.layers_bn[2*i](self.layers_conv[2*i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2*i + 1](self.layers_conv[2*i + 1](x))))

        x = self.shrink(x)
        return x

if __name__ == '__main__':
    tm  = TemporalModel(512,filter_widths=[3,3],fc_summon=True,fc_dim=512).cuda()
    x=torch.rand(2,9,512).cuda()
    fcc=torch.rand(2,512).cuda()
    y = tm(x,fcc=fcc)
    print('Pass')