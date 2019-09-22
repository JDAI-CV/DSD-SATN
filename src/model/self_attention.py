import torch.nn as nn
import torch
import os
import sys
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('model/',''))
from utils.SMPL import SMPL
from utils.util import *
from transformer.Models import *
import torch.nn.functional as F
from model.temporal_model import *

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, batch_size=16, spwan=9, d_word_vec=512,
            n_layers=2, n_head=8, d_k=64, d_v=64,
            d_model=512, d_inner=2048, dropout=0.1,no_positional_coding =False):

        super().__init__()
        self.span =spwan
        self.batch_size = batch_size
        self.no_positional_coding = no_positional_coding

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.slf_attn_mask = torch.zeros((batch_size,self.span,self.span)).byte().cuda()
        self.non_pad_mask = torch.ones((batch_size,self.span,1)).float().cuda()
        if not self.no_positional_coding:
            self.position_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(spwan+1, d_word_vec, padding_idx=0),freeze=True).cuda()
            self.position_encoding = self.position_enc(torch.stack([torch.arange(self.span) for i in range(self.batch_size)]).cuda())

        print('Created Transform Encoder')

    def forward(self, enc_output, return_attns=False,pcoding=None):
        enc_slf_attn_list = []

        if pcoding is None and not self.no_positional_coding:
            pcoding = self.position_encoding

        if self.slf_attn_mask.shape[0]!=enc_output.shape[0]:
            self.slf_attn_mask = torch.zeros((enc_output.shape[0],self.span,self.span)).byte().cuda()
            self.non_pad_mask = torch.ones((enc_output.shape[0],self.span,1)).float().cuda()
            if not self.no_positional_coding:
                self.position_encoding = self.position_enc(torch.stack([torch.arange(self.span) for i in range(enc_output.shape[0])]).cuda())
        #print('ecoding shape', pcoding.shape, enc_output.shape)
        if not self.no_positional_coding:
            enc_output = enc_output + pcoding

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=self.non_pad_mask,
                slf_attn_mask=self.slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
            #print('enc_output,enc_slf_attn:',enc_output.shape,enc_slf_attn.shape)#[64, 24, 512] [512, 24, 24]
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output



class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,smooth_loss = False, batch_size=16,spwan=9,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=2, n_head=8, d_k=64, d_v=64, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,cross_entropy_loss = True,
            emb_src_tgt_weight_sharing=True,no_positional_coding =False,without_loss = False,
            big_sorting_layer=False, fusion_type='bilinear',fc_summon=False,fc_dim=512):

        super().__init__()
        self.batch_size = batch_size
        self.span = spwan
        self.no_positional_coding = no_positional_coding
        self.smooth_loss = smooth_loss
        self.without_loss = without_loss
        self.cross_entropy_loss = cross_entropy_loss
        self.big_sorting_layer = big_sorting_layer

        if self.span==9:
            filter_widths = [3,3]
        elif self.span==27:
            filter_widths = [3,3,3]
        elif self.span ==3:
            filter_widths = [3]
        elif self.span ==81:
            filter_widths = [3,3,3,3]

        self.convs2s = TemporalModel(d_model,filter_widths=filter_widths,fc_summon=fc_summon,fc_dim=fc_dim)

        if d_model!=512:
            self.converter = nn.Linear(d_model, 512, bias=False)
            d_model = 512
        d_word_vec = d_model

        self.encoder = Encoder(batch_size=batch_size, spwan=self.span,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout,no_positional_coding =no_positional_coding)

        self.tgt_word_prj = nn.Linear(d_model, self.span, bias=False)
        nn.init.kaiming_normal_(self.tgt_word_prj.weight, mode='fan_out', nonlinearity='relu')

        if not self.no_positional_coding:
            self.position_enc = nn.Embedding.from_pretrained(
                get_sinusoid_encoding_table(self.span+1, d_word_vec, padding_idx=0),freeze=True).cuda()
            self.position_encoding = self.position_enc(torch.stack([torch.arange(self.span) for i in range(self.batch_size)]).cuda())

    def forward(self, enc_in,target=None, shuffle=False,shuffle_ratio=0.5,kp3d_24=False, pcoding=None):
        if enc_in.shape[-1]!=512:
            enc_in = self.converter(enc_in)

        if shuffle:
            random_index = torch.stack([torch.randperm(self.span) for i in range(enc_in.shape[0])]).cuda()
            if not self.no_positional_coding:
                pcoding = self.position_enc(random_index)
                for idx,random_index_row in enumerate(random_index):
                    enc_in[idx] = enc_in[idx,random_index_row]
            reorder_index = self.reorder(random_index,only_idx=True)
        else:
            random_index = torch.stack([torch.arange(self.span) for i in range(enc_in.shape[0])]).cuda()
            reorder_index = random_index
            if not self.no_positional_coding:
                pcoding = self.position_encoding
                if pcoding.shape[0]!=enc_in.shape[0]:
                    pcoding = self.position_enc(random_index)

        enc_output = self.encoder(enc_in, pcoding=pcoding)

        reordered_output = self.reorder(random_index,enc_output.clone()) if shuffle else enc_output

        out = self.convs2s(reordered_output, just_out=True,kp3d_24=kp3d_24,fcc=enc_in[:,self.span//2])
        if not shuffle:
            return out

        if self.big_sorting_layer:
            enc_output = enc_output.permute((0,2,1))
            enc_output = self.convert_conv(enc_output)
            enc_output = enc_output.permute((0,2,1))

        seq_logit = self.tgt_word_prj(enc_output)

        if self.without_loss:
            return out, 0.
        loss = self.cal_loss(seq_logit,reorder_index,smooth=self.smooth_loss,cross_entropy=self.cross_entropy_loss)

        return out,loss

    def reorder(self, random_index, y=None, only_idx = False):

        random_index = random_index.cpu().numpy()
        reorder_index = np.zeros_like(random_index)
        for i in range(random_index.shape[1]):
            x_idx,y_idx =np.where(random_index==i)
            reorder_index[x_idx,i] = y_idx
        if only_idx:
            return torch.from_numpy(reorder_index).cuda()
        for idx,reorder_index_row in enumerate(reorder_index):
            y[idx] = y[idx,reorder_index_row]
        return y


    def cal_loss(self, output, target_idx, smooth=False,cross_entropy=True):
        batch_size, spwan, n_class = output.shape

        output = output.view(-1, output.size(2))
        if cross_entropy:
            output = F.log_softmax(output, dim=1)
        else:
            output = F.softmax(output, dim=1)

        eps = 0.1
        one_hot = torch.zeros_like(output).scatter(1, target_idx.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        if smooth:
            smooth_factor=0.3
            one_hot = one_hot.reshape(batch_size, spwan, n_class)
            new_one_hot = torch.zeros_like(one_hot)
            for i in range(spwan):
                fore_frame = i-1
                next_frame = i+1
                if fore_frame<0:
                    fore_frame=0
                if next_frame>spwan-1:
                    next_frame= spwan-1
                new_one_hot[:,i] = (smooth_factor * one_hot[:,fore_frame] + one_hot[:,i] + smooth_factor * one_hot[:,next_frame])/(1+2*smooth_factor)
            one_hot = new_one_hot.view(-1,n_class)
            one_hot_reverse = torch.from_numpy(np.ascontiguousarray(one_hot.cpu().numpy()[:,::-1])).cuda()

        target = target_idx.contiguous().view(-1)
        if cross_entropy:
            loss = -(one_hot * output).sum(dim=1).sum()
        else:
            loss = torch.sqrt(((one_hot - output)**2).sum(dim=1)).sum()
        if smooth:
            if cross_entropy:
                loss_reverse = -(one_hot_reverse * output).sum(dim=1).sum()
            else:
                loss_reverse = torch.sqrt(((one_hot_reverse - output)**2).sum(dim=1)).sum()
            if loss_reverse.item()<loss.item():
                return loss_reverse
        return loss


if __name__ == '__main__':
    tm = Transformer(structure = True, structure_type='learnable_pooling',
            fusion_type='fc_summon',fc_summon=True,fc_dim=512).cuda()
    for i in range(20):
        x=torch.rand(2,243,512).cuda()
        y = tm(x)