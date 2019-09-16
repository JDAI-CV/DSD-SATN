
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
import time
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from utils.util import *
import config
from config import args
from utils.SMPL import SMPL
from dataset.h36m_dataset import *
import copy
import utils.neuralrenderer_render as neuralrenderer
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

class hum36m_dataloader(Human36mDataset):
    def __init__(self, scale_range, train_flag=True):
        self.data_folder = config.data_set_path['h36m']#os.path.join(args.dataset_rootdir,'h36m/')
        print('Dataset location of human3.6M dataset:',self.data_folder)
        self.image_folder = os.path.join(self.data_folder,'images/')

        self.annots_file = os.path.join(self.data_folder,'annots.npz')
        self.scale_range = scale_range
        self.pix_format = args.pix_format
        self.normalize = args.normalize
        self.train_flag=train_flag
        self.test_flag=1-train_flag
        self.train_test_subject = {'train':['S1','S5','S6','S7','S8'],'test':['S9','S11']}
        self.h36m32_2_lsp14 = [3,2,1,6,7,8,27,26,25,17,18,19,13,15]
        self.coco18_2_lsp14 = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 5, 6, 1, 0]
        self.mpii_2_lsp14 = [0,1,2,3,4,5, 10,11,12,13,14,15, 8,9]
        self.crop_size = 256
        self.sigma=2.
        self.video=args.video
        self.eval_mode = args.eval_mode
        self.high_resolution=args.high_resolution

        self.video_frame_spawn = args.receptive_field
        self.with_kps = args.with_kps
        self.save_cropped = False
        self.augment_half = True
        phase = 'train' if self.train_flag else 'test'
        if self.with_kps:
            self.kps_alpha_format = args.alpha_format
            self.kps_alpha_json_file = os.path.join(self.data_folder,"h36m_{}_{}_keypoints_alphapose_result.json".format(phase,self.kps_alpha_format))
        self.imgs_list_file = os.path.join(self.data_folder,"h36m_{}.txt".format(phase))
        self.video_features_npz = os.path.join(self.data_folder,"annots_{}_bilinearf.npz".format(phase))

        print('Start loading human3.6m data.')
        self.load_file_list()

        if args.save_features or self.video:
            self.train_flag=False
        if not self.train_flag:
            self.scale_range=[1.4,1.6]
        print('Loaded files ,total {} samples'.format(len(self.file_paths)))
        if self.video:
            self.feature_name = args.feature_name
            self.load_video_file_list()
            print('Loaded video clips ,total {} samples'.format(len(self.video_paths)))

    def load_video_file_list(self):
        self.annots_bf = np.load(self.video_features_npz, allow_pickle=True)['annots'][()]
        self.video_dict = {}

        for name in self.annots_bf.keys():
            subject, active, cam, frame_name = name.split('_')
            video_name = '{}_{}_{}'.format(subject, active, cam)
            if video_name not in self.video_dict:
                self.video_dict[video_name] = []
            self.video_dict[video_name].append(int(frame_name.split('.jpg')[0]))
        for video_name in self.video_dict.keys():
            self.video_dict[video_name] = np.array(sorted(self.video_dict[video_name]))

        self.video_clips_idx = {}
        self.video_count = 0
        spawn = int((self.video_frame_spawn-1)/2)
        for video_name in self.video_dict.keys():
            bf_list = self.video_dict[video_name]
            self.video_clips_idx[video_name] = {}
            total_frame = len(bf_list)
            for current_frame in range(total_frame):
                features_idx = np.arange(current_frame-spawn, current_frame+spawn+1)
                features_idx[features_idx<0] = 0
                features_idx[features_idx>=total_frame] = total_frame-1
                self.video_clips_idx[video_name][current_frame] = bf_list[features_idx]
                self.video_count += 1

        self.video_paths = []
        self.video_count = {}
        for file in self.file_paths:
            video_name = file.rstrip(file.split('_')[-1])
            if video_name in self.video_count:
                self.video_count[video_name] += 1
            else:
                self.video_count[video_name] = 1
        spawn = int((self.video_frame_spawn-1)/2)
        for video_path, count_num in self.video_count.items():
            for start_frame in range(count_num):
                self.video_paths.append([video_path,start_frame,count_num])


    def load_file_list(self):
        self.file_paths = []
        self.annots = np.load(self.annots_file, allow_pickle=True)['annots'][()]

        with open(self.imgs_list_file) as f:
            test_list = f.readlines()
        for test_file in test_list:
            self.file_paths.append(test_file.strip())

        if self.with_kps:
            self.kps_alphas = {}
            empty_kps = np.zeros((14,3))
            with open(self.kps_alpha_json_file,'r') as f:
                raw_labels = json.load(f)
            frame_num = len(raw_labels)
            print('frame_num',frame_num)
            error_count=0
            for j,img_name in enumerate(raw_labels.keys()):
                if self.kps_alpha_format=='coco':
                    img_name = os.path.basename(self.file_paths[j])
                    try:
                        kp2d = np.array(raw_labels[img_name]['bodies'][0]['joints']).reshape(-1,3)[self.coco18_2_lsp14]
                        kp2d[-1,2]=0
                        kp2d[:,2] = kp2d[:,2]>0
                    except:
                        kp2d = empty_kps
                        error_count+=1

                elif self.kps_alpha_format == 'mpii':
                    content = raw_labels[img_name]
                    poses = []
                    for pid in range(len(content)):
                        poses.append(np.array(content[pid]['keypoints']).reshape(-1,3)[:,:3])
                    poses = np.array(poses)
                    pid_best = np.argmax(np.array(poses[:,:,2]),0)
                    pose = np.array(poses[pid_best,np.arange(poses.shape[1])])
                    kp2d = pose[self.mpii_2_lsp14]

                self.kps_alphas[img_name] = kp2d
            if self.eval_mode=='frontal_only':
                for f in self.file_paths:
                    if '_3_' not in f:
                        self.file_paths.remove(f)

            print('error_count:',error_count)
            print('kps image example:',list(self.kps_alphas.keys())[:2])

    def generate_index_map(self):
        self.index_map = [0]
        for idx, info in enumerate(self.h5_file_list):
            self.index_map.append(self.index_map[idx]+info['kp3d'].shape[0])
        self.index_map = np.array(self.index_map)

    def get_image_info(self,imgname,total_frame=None):
        info = self.annots[imgname]
        camkp3d = info['kp3d_mono']
        kp3d_mono = align_by_pelvis_single(camkp3d[self.h36m32_2_lsp14])

        imgpath = os.path.join(self.image_folder,imgname)
        frame = cv2.imread(imgpath)[:,:,::-1]
        kp2d = info['kp2d'][self.h36m32_2_lsp14].astype(np.float32)

        if self.with_kps:
            try:
                kps_alpha = self.kps_alphas[imgname]
                fail_flag=False
            except Exception as error:
                fail_flag=True

        if not self.with_kps or fail_flag:
            kps_alpha = np.concatenate([kp2d,np.ones((14,1))],1)
            kps_alpha[-1,2] = 0

        kp3d = info['kp3d'][self.h36m32_2_lsp14]
        smpl_randidx = random.randint(0,2)
        root_rotation = np.array(info['cam'])[smpl_randidx]
        pose = np.array(info['poses'])[smpl_randidx]
        global_rotation = copy.deepcopy(pose[:3])
        pose[:3] = root_rotation
        smpl = np.concatenate([info['trans'][smpl_randidx],\
         pose, info['betas']])

        return frame, kp2d,kps_alpha, kp3d, kp3d_mono, smpl,global_rotation, imgpath

    def get_gt_params(self,video_path,frame_idxs):
        params = []
        for idx in frame_idxs:
            info = np.load('{}{}.npz'.format(video_path,idx), allow_pickle=True)
            pose = np.array(info['poses'])[1]
            root_rotation = np.array(info['cam'])[1]
            pose[:3] = root_rotation
            smpl = np.concatenate([pose, info['betas']])
            params.append(smpl)
        return np.array(params)

    def get_item_video(self,index):
        video_path, current_frame, total_frame = self.video_paths[index]
        video_name = os.path.basename(video_path)
        imgpath = video_path+'{}.jpg'.format(current_frame)
        video_input = self.get_item_single_frame(imgpath,total_frame=total_frame)

        features_idx = []
        current_spawn = int((self.video_frame_spawn-1)/2)
        for num in range(current_frame-current_spawn, current_frame+current_spawn+1):
            if num<0:
                num=0
            elif num>=total_frame:
                num = total_frame-1
            features_idx.append(num)

        bilinear_feature = np.array([self.annots_bf['{}{}.jpg'.format(video_name,i)] for i in features_idx],dtype=np.float32)
        video_input.update({'image':torch.from_numpy(bilinear_feature).float()})

        return video_input

    def process_kps(self,kps,image):
        ratio_w = 1.0 * self.crop_size / image.shape[0]
        kps[:,0] *= ratio_w
        ratio_h = 1.0 * self.crop_size / image.shape[1]
        kps[:,1] *= ratio_h

        ratio = 1.0 / self.crop_size
        kps[:,:2] = 2.0 * kps[:,:2] * ratio - 1.0

        if kps.shape[1]>2:
            kps[kps[:,2]<0.01,:2] = -2.
            kps=kps[:,:2]
        return kps

    def get_item_single_frame(self,imgname,total_frame=None):
        image, kps,kps_alpha, kp3d, kp3d_mono, param, global_rotation, imgpath = self.get_image_info(imgname,total_frame=total_frame)

        scale = np.random.rand(4) * (self.scale_range[1] - self.scale_range[0]) + self.scale_range[0] if self.train_flag else self.scale_range[0]
        box = calc_aabb(kps)

        if self.train_flag and self.augment_half and random.random()<0.2:
            scale=random.random()*0.6+0.4
            box[0]-=np.array([0,np.random.randint(80,140)])

        image, kps, kps_offset = cut_image(image, kps, scale, box[0], box[1])
        offset,lt,rb,size,_ = kps_offset
        leftTop = [lt[0]-offset[0],lt[1]-offset[1]]
        if self.with_kps:
            kps_alpha = off_set_pts(kps_alpha, leftTop)
            kps_alpha = self.process_kps(kps_alpha,image)
        offsets = np.array([image.shape[1],image.shape[0],lt[1],rb[1],lt[0],rb[0],offset[1],size[1],offset[0],size[0]],dtype=np.int)
        kps = self.process_kps(kps,image)

        dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation = cv2.INTER_CUBIC)
        org_image = cv2.resize(image, (500,500), interpolation = cv2.INTER_CUBIC) if self.high_resolution else dst_image.copy()

        input_data = {
            'image_org': torch.from_numpy(org_image),
            'kp_2d': torch.from_numpy(kps).float(),
            'kp_3d': torch.from_numpy(kp3d).float(),
            'kp3d_mono': torch.from_numpy(kp3d_mono).float(),
            'param': torch.from_numpy(param).float(),
            'global_rotation': torch.from_numpy(global_rotation).float(),
            'imgpath': imgpath,
            'offsets': torch.from_numpy(offsets),
            'result_dir': imgpath,
            'name':imgpath,
            'data_set':'h36m',}
        if self.with_kps:
            input_data.update({'kps_alpha': torch.from_numpy(kps_alpha).float()})
        if not self.video:
            image_input = {'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, self.pix_format, self.normalize)).float(),}
            input_data.update(image_input)

        return input_data

    def __len__(self):
        if self.video:
            return len(self.video_paths)
        return len(self.file_paths)

    def __getitem__(self, index):
        if self.video:
            return self.get_item_video(index)
        else:
            imgname = self.file_paths[index]
            return self.get_item_single_frame(imgname)

        return video_input

    def summum_features(self):
        print('All video num: ',len(self.video_count))
        for idx, (video_path, frame_num) in enumerate(self.video_count.items()):
            if idx%10==0:
                print(idx)
            bilinear_features = []
            backbone_features = []
            for num in range(frame_num):
                npzpath = video_path+'{}_cropped_features_kp2d_best.npz'.format(num)
                data = np.load(npzpath, allow_pickle=True)
                bilinear_features.append(data['bilinear'])
                #backbone_features.append(data['backbone'])
                backbone_features.append(data['params'])
            bilinear = np.array(bilinear_features)
            bilinear_name = video_path+'cropped_bilinear_kp2d_features_best.npy'
            backbone = np.array(backbone_features)
            #backbone_name = video_path+'cropped_backbone_features_kp3d.npy'
            backbone_name = video_path+'cropped_params_kp2d_features_best.npy'
            np.save(bilinear_name,bilinear)
            np.save(backbone_name,backbone)

    def crop_restframe_func(self,imgpath):
        frame = cv2.imread(imgpath)[:,:,::-1]
        cropped_img_kps_path = imgpath.replace('h36m_restframe','h36m').split('_'+imgpath.split('_')[-1])[0]+'.npz'
        if not os.path.isfile(cropped_img_kps_path):
            video_path_miss = '{}_{}_{}'.format(*os.path.basename(imgpath).split('_')[:3])
            if video_path_miss not in self.lack_video:
                self.lack_video.append(video_path_miss)
            return None, None, False

        kp2d = np.load(cropped_img_kps_path, allow_pickle=True)['kp2d'][self.h36m32_2_lsp14].astype(np.float32)
        return frame,kp2d,True

if __name__ == '__main__':
    import time
    data_time = AverageMeter()
    h36m = hum36m_dataloader([1.0, 1.6], True)
    dataloader = DataLoader(
            dataset = h36m,
            batch_size = 16,
            shuffle = True,
            drop_last = False,
            pin_memory = True,
            num_workers = 2)
    device = torch.device('cuda', 1)
    #smpl = SMPL(args.smpl_model, joint_type = 'lsp', obj_saveable = True).to(device)
    os.makedirs('./test',exist_ok=True)
    end_time = time.time()
    for _,r in enumerate(dataloader):
        data_time.update(time.time() - end_time)
        end_time = time.time()
        if _%100==0:
            print(_)
            print('{:.3f}'.format(data_time.avg))
            data_time = AverageMeter()

        #continue
        if _%100==0:
            for key,value in r.items():
                if isinstance(value,torch.Tensor):
                    print(key,value.shape)
                elif isinstance(value,list):
                    print(key,len(value))

        #image_aug = (r['image'][0].numpy()).astype(np.uint8)[:,:,::-1]
        #cv2.imwrite('./test/augimage_{}.png'.format(_), image_aug)
        image = r['image_org'][0].numpy().astype(np.uint8)[:,:,::-1]
        #image = (r['dst_mask'][1]*255).numpy().astype(np.uint8)
        #image = np.stack((image,image,image),axis=2)
        kps = r['kp_2d'][0].numpy()
        kps = r['kps_alpha'][0].numpy()
        kps = (kps + 1) * 256 / 2.0
        #base_name = os.path.basename(r['image_name'])
        image_real = draw_lsp_14kp__bone(image.copy(), kps)
        cv2.imwrite('./test/kp_{}.png'.format(_), image_real)

        continue

        param = r['param'][0]
        cam = param[:3]
        pose_g = param[3:75].unsqueeze(0)
        shape_g = param[75:].unsqueeze(0)

        verts,joints,Rs = smpl(shape_g.to(device),pose_g.to(device), get_skin = True)
        kp3d_mono = r['kp3d_mono'][0].reshape(14,3).cuda()
        predicts_aligned = align_by_pelvis_single(joints[0].cpu().numpy())
        mpjpe_each = torch.sqrt(((predicts_aligned - kp3d_mono)**2).sum(-1)).mean(-1)*1000

        show3Dpose([ r['kp3d_mono'][0].numpy(),predicts_aligned], lcolor = ['r','g'],rcolor = ['b','black'], save_path='./test/camkp3d_{}.png'.format(_))
