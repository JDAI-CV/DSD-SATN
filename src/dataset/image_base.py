
import sys
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import shutil
import time
import pickle
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from utils.util import *
import config
from config import args
import copy
import scipy.io as scio

class Image_base(object):
    def __init__(self, train_flag=True,high_resolution=False):
        self.data_folder = args.dataset_rootdir
        self.scale_range = [1.1,1.1]
        self.use_flip = False
        self.flip_prob = 0.5
        self.normalize = True
        self.augment_half = False
        self.pix_format = "NCHW"
        self.train_flag=train_flag
        self.h36m32_2_lsp14 = [3,2,1,6,7,8,27,26,25,17,18,19,13,15]
        self.coco18_2_lsp14 = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 5, 6, 1, 0]
        self.coco25_2_lsp14 = [11, 10, 9, 12, 13, 14, 4, 3, 2, 5, 5, 6, 1, 0]
        self.mpii_2_lsp14 = [0,1,2,3,4,5, 10,11,12,13,14,15, 8,9]
        self.crop_size = 256
        self.high_resolution=high_resolution
        self.labels = []
        self.images = []
        self.file_paths = []
        self.get_empty_input()

    def get_image_info(self,index):
        raise NotImplementedError

    def get_empty_input(self):
        self.empty_kp3d = torch.zeros(14,3).float()
        self.empty_param = torch.zeros(85).float()
        self.empty_gr = torch.zeros(3).float()

    def process_kps(self,kps,image):
        ratio_w = 1.0 * self.crop_size / image.shape[0]
        kps[:,0] *= ratio_w
        ratio_h = 1.0 * self.crop_size / image.shape[1]
        kps[:,1] *= ratio_h

        ratio = 1.0 / self.crop_size
        kps[:,:2] = 2.0 * kps[:,:2] * ratio - 1.0
        if kps.shape[1]>2:
            kps[kps[:,2]<1,:2] = -2.
            kps=kps[:,:2]
        return kps

    def get_item_single_frame(self,index):

        image, kps, box,metas = self.get_image_info(index)
        dataset_name, imgpath,result_dir, kp3d, kp3d_mono,param, global_rotation = metas

        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0] if self.train_flag else self.scale_range[0]
        if self.train_flag and self.augment_half and random.random()<0.2:
            scale=random.random()*0.6+0.4
            box[0]-=np.array([0,np.random.randint(80,140)])

        image, kps, kps_offset = cut_image(image, kps, scale, box[0], box[1])
        if isinstance(image,bool):
            return self.__getitem__(abs(index-1))#(np.random.randint(len(self.file_paths)))
        offset,lt,rb,size,_ = kps_offset
        offsets = np.array([image.shape[1],image.shape[0],lt[1],rb[1],lt[0],rb[0],offset[1],size[1],offset[0],size[0]],dtype=np.int)

        if self.train_flag and self.use_flip and random.random() <= self.flip_prob:
            image, kps = flip_image(image, kps)

        dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation = cv2.INTER_CUBIC)
        org_image = cv2.resize(image, (500,500), interpolation = cv2.INTER_CUBIC) if self.high_resolution else dst_image.copy()

        kps = self.process_kps(kps,image)

        input_data = {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),
            'image_org': torch.from_numpy(org_image),
            'kp_2d': torch.from_numpy(kps).float(),
            'kp_3d': kp3d,
            'kp3d_mono': kp3d_mono,
            'param': param,
            'global_rotation': global_rotation,
            'imgpath': imgpath,
            'offsets': torch.from_numpy(offsets),
            'result_dir': result_dir,
            'name':imgpath,
            'data_set':dataset_name}
        if args.with_kps:
            kps_alpha = kps.copy()
            input_data.update({'kps_alpha': torch.from_numpy(kps_alpha).float(),})

        return input_data

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        return self.get_item_single_frame(index)

    def read_pkl(self,file_path):
        return pickle.load(open(file_path,'rb'),encoding='iso-8859-1')

    def read_json(self,file_path):
        with open(file_path,'r') as f:
            file = json.load(f)
        return file

    def read_npy(self,file_path):
        return np.load(file_path)

def visualize_renderer(verts,images,renderer):
    images = cv2.resize(images.astype(np.uint8),(224,224))
    renders = (renderer.forward(verts).detach().cpu().numpy().transpose((0,2,3,1))*256).astype(np.uint8)
    render_mask = ~(renders.astype(np.bool))
    for i in range(render_mask.shape[0]):
        renders[i][render_mask[i]] = images[render_mask[i]]
    return renders

def check_and_mkdir(dir):
    os.makedirs(dir,exist_ok=True)

def test_dataset(dataset,with_3d=False):
    save_dir = 'test'
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    data_time = AverageMeter()
    print('Initialized dataset')

    dataloader = DataLoader(dataset = dataset,batch_size = 16,shuffle = True,\
        drop_last = False,pin_memory = True,num_workers = 6)

    if with_3d:
        from utils.SMPL import SMPL
        import utils.neuralrenderer_render as nr
        device = torch.device('cuda', 1)
        smpl = SMPL(args.smpl_model, joint_type = 'lsp', obj_saveable = True).to(device)
        renderer = nr.get_renderer().to(device)
    end_time = time.time()
    for _,r in enumerate(dataloader):
        data_time.update(time.time() - end_time)
        end_time = time.time()
        if _%100==0:
            print(_)
            print('{:.3f}'.format(data_time.avg))
            data_time = AverageMeter()

        if _%100==0:
            for key,value in r.items():
                if isinstance(value,torch.Tensor):
                    print(key,value.shape)
                elif isinstance(value,list):
                    print(key,len(value))

        image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
        kps = r['kp_2d'][0].numpy()
        #kps = r['kps_alpha'][0].numpy()
        kps = (kps + 1) * 256 / 2.0
        image_real = draw_lsp_14kp__bone(image.copy(), kps)
        cv2.imwrite('./{}/kp_{}.png'.format(save_dir,_), image_real)

        if with_3d:
            param = r['param'][0]
            pose_g = param[3:75].unsqueeze(0).float()
            shape_g = param[75:].unsqueeze(0).float()
            verts,joints,Rs = smpl(shape_g.to(device),pose_g.to(device), get_skin = True)
            smpl.save_obj(verts.cpu().numpy()[0],'./test/obj_{}_{}.obj'.format(_,0))
