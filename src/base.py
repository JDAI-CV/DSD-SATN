import time
import sys
import os
import cv2
import copy
import random
import datetime
import numpy as np
from prettytable import PrettyTable
from collections import OrderedDict

import config
from config import args
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from dataset.human36m_dataset import hum36m_dataloader
from dataset.mpii import MPIIDataset
from dataset.AICH import AICH
from dataset.up import UP
from dataset.pw3d import PW3D
from dataset.mosh import Mosh

from model.model import get_pose_net
from model.discriminator import Discriminator
from model.temporal_model import TemporalModel
from model.self_attention import Transformer

from utils.util import *
from scipy.spatial import procrustes
from utils.evaluation_matrix import *
from utils.eval_pckh import *
from utils.SMPL import SMPL


class Base(object):
    def __init__(self):
        self.load_config_dict(vars(args))
        self.log_file = os.path.join(self.log_path,'{}.log'.format(self.tab))
        self.write2log('================ Training Loss (%s) ================\n' % time.strftime("%c"))

        self.result_img_dir = '../result_image/{}_on_gpu{}'.format(self.tab,self.gpu)
        if not os.path.isdir(self.result_img_dir):
            os.makedirs(self.result_img_dir)

        self.adjust_lr_epoch = [int(self.epoch*0.8),int(self.epoch*0.95)]

    def _build_model(self):
        print('start building modle.')
        if self.video:
            generator = Transformer(smooth_loss =self.shuffle_smooth_loss,batch_size=self.batch_size,d_model=self.features_channels, spwan=self.receptive_field, n_layers=self.attention_layer, n_head=8,big_sorting_layer = False,\
                no_positional_coding =False,without_loss = 1-self.shuffle_aug, cross_entropy_loss=True)
            print('created video model.')
        else:
            generator = get_pose_net()
            print('created single-frame model.')

        if self.fine_tune or self.eval:
            generator = self.load_model(self.gmodel_path,generator)

        self.generator = nn.DataParallel(generator).cuda()
        self.optimizer = torch.optim.SGD(self.generator.parameters(), lr = self.lr, momentum = 0.9)
        self.e_sche = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones = self.adjust_lr_epoch, gamma = self.adjust_lr_factor)

        if not self.eval:
            discriminator = Discriminator()
            if self.fine_tune:
                discriminator = self.load_model(self.dmodel_path,discriminator)
            self.discriminator = nn.DataParallel(discriminator).cuda()
            self.d_opt = torch.optim.Adam(self.discriminator.parameters(),lr = self.lr)
            self.d_sche = torch.optim.lr_scheduler.MultiStepLR(self.d_opt, milestones = self.adjust_lr_epoch, gamma = self.adjust_lr_factor)

        if self.eval_with_single_frame_network:
            spatial_feature_extractor = get_pose_net()
            model_path_single_frame = config.best_model_dict['dsd_single']
            spatial_feature_extractor = self.load_model(model_path_single_frame,spatial_feature_extractor)
            self.spatial_feature_extractor = nn.DataParallel(spatial_feature_extractor).cuda()

        self.smpl = SMPL(self.smpl_model, joint_type = 'lsp', obj_saveable = True).cuda()

        print('finished build model.')


    def _create_data_loader(self,train_flag=True,hard_minging=False):
        print('gathering datasets')
        if self.internet:
            datasets = Internet(train_flag = train_flag,high_resolution = self.high_resolution, spawn = self.receptive_field,video=self.video)
        elif self.test_single:
            datasets = Deepfashion(train_flag = train_flag,high_resolution = self.high_resolution)#MPV(train_flag = train_flag,high_resolution = self.high_resolution)#
        elif self.eval_pw3d:
            datasets = PW3D(train_flag = train_flag,high_resolution = self.high_resolution, spawn = self.receptive_field,video=self.video,kps_alpha_format=self.alpha_format)
        else:
            datasets_list = []
            if self.with_h36m:
                h36m = hum36m_dataloader(scale_range = [1.0, 1.6],train_flag=train_flag)#[1.4, 1.6],
                datasets_list = [h36m]
            if self.with_up:
                updataset = UP(train_flag=train_flag,high_resolution=self.high_resolution)
                datasets_list.append(updataset)
            if self.with_mpii:
                mpii = MPIIDataset(train_flag=train_flag,high_resolution=self.high_resolution,)
                datasets_list.append(mpii)
            if self.with_aich:
                aich = AICH(train_flag=train_flag,high_resolution=self.high_resolution,)
                datasets_list.append(aich)
            if self.with_pa:
                pa = Penn_Action(train_flag = train_flag,high_resolution = self.high_resolution,kps_alpha_format=self.alpha_format,spawn = self.receptive_field,video=self.video,receptive_field = self.receptive_field,)
                datasets_list.append(pa)

            datasets = torch.utils.data.ConcatDataset(list(datasets_list))
        print('gathered datasets')

        return DataLoader(dataset = datasets, batch_size = self.batch_size if train_flag else self.val_batch_size,\
            shuffle = True,drop_last = False, pin_memory = True,num_workers = self.nw)

    def _create_adv_data_loader(self, data_adv_set):
        data_set = []
        for data_set_name in data_adv_set:
            data_set_path = config.data_set_path[data_set_name]
            if data_set_name == 'mosh':
                mosh = Mosh(data_set_path = data_set_path,)
                data_set.append(mosh)
            else:
                msg = 'invalid adv dataset'
                sys.exit(msg)

        con_adv_dataset = ConcatDataset(data_set)
        return DataLoader(dataset = con_adv_dataset,batch_size = self.batch_size, shuffle = True,drop_last = True,pin_memory = True)

    def net_forward(self,data_3d,model,only_blf=False,video=False,additional_data=None):
        imgs = data_3d['image'].cuda()
        if self.with_kps:
            additional_data = data_3d[self.kps_type].clone().cuda()
            additional_data = additional_data.reshape(additional_data.shape[0],-1)
            
        if video:
            if not self.shuffle_aug:
                outputs = model(imgs, additional_data, shuffle=self.shuffle_aug,kp3d_24=self.kp3d_24)
                return outputs, outputs[2], None
            outputs, loss = model(imgs, additional_data, shuffle=self.shuffle_aug,kp3d_24=self.kp3d_24)
            return outputs, outputs[2], loss
        else:
            if only_blf:
                return model(imgs,kps_gt=additional_data,kp3d_24=self.kp3d_24,only_blf=only_blf)
            outputs,kps,details = model(imgs,kps_gt=additional_data,kp3d_24=self.kp3d_24)
        return outputs,kps,details

    def save_single_model(self, model, path):
        print('saving ',path)
        model_save = model.state_dict()
        torch.save(model_save, path)

    def save_model(self,title,parent_folder ='./trained_models'):

        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)

        generator_save_path = os.path.join(parent_folder, title + '.pkl')
        self.save_single_model(self.generator, generator_save_path)

        disc_save_path = os.path.join(parent_folder, 'D_' + title + '.pkl')
        self.save_single_model(self.discriminator, disc_save_path)

        return generator_save_path

    def print_net(self,model,name):
        print(name,'requires_grad')
        states = []
        for param in model.parameters():
            if not param.requires_grad:
                states.append(param.name)
        if len(states)<1:
            print('All parameters are trainable.')
        else:
            print(states)

    def write2log(self,massage):
        with open(self.log_file, "a") as log_file:
            log_file.write(massage)

    def process_pretrained(self,model_dict):
        keys = list(model_dict.keys())
        for key in keys:
            if 'module.net.features' in key:
                num = int(key.split('.')[-2])
                if num==0:
                    continue
                type_name = key.split('.')[-1]
                model_dict['module.net.features.'+str(num+1)+'.'+type_name] = model_dict[key]
        return model_dict

    def get_only_h36m_data(self,data_3d,name='h36m'):
        data_3d['data_set'] = np.array(data_3d['data_set'])
        data_3d['imgpath'] = np.array(data_3d['imgpath'])
        h36m_idx = np.where(data_3d['data_set']==name)[0].astype(np.int)
        if len(h36m_idx)==0:
            return data_3d,False
        for key in data_3d.keys():
            data_3d[key] = data_3d[key][h36m_idx]
        return data_3d,True


    def load_data_iter_format(self,dataloader):
        try:
            data = next(dataloader)
        except StopIteration:
            dataloader = iter(self.loader_disc)
            data = next(dataloader)
        return data


    def h36m_evaluation_act_wise(self,results,imgpaths):
        actions = []
        action_results = []
        for imgpath in imgpaths:
            actions.append(os.path.basename(imgpath).split('.jpg')[0].split('_')[1].split(' ')[0])
        for action_name in self.action_names:
            action_idx = np.where(np.array(actions)==action_name)[0]
            action_results.append('{:.2f}'.format(results[action_idx].mean()))
        return action_results

    def load_config_dict(self, config_dict):
        for i, j in config_dict.items():
            setattr(self,i,j)

    def load_model(self, path, model,prefix = 'module.'):
        print('*'*20)
        print('using fine_tune model: ', path)
        if os.path.exists(path):
            copy_state_dict(model.state_dict(), torch.load(path), prefix = prefix)
        else:
            print('model {} not exist!'.format(path))
        print('*'*20)
        return model
