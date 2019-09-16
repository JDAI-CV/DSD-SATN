import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from dataset.video_base import *

class PW3D(Video_base):
    def __init__(self,train_flag = False,high_resolution = False, spawn = 9,video=False, get_features=False,feature_name='bilinear',receptive_field = 81,kps_alpha_format='mpii'):
        super(PW3D,self).__init__(train_flag,video,get_features,feature_name,receptive_field,spawn,high_resolution)
        self.data_folder = config.data_set_path['pw3d']#os.path.join(self.data_folder,'3DPW/')
        self.data3d_dir = os.path.join(self.data_folder,'sequenceFiles')
        self.image_dir = os.path.join(self.data_folder,'imageFiles')
        self.kps_alpha_json_file = os.path.join(self.data_folder,'3dpw_{}_keypoints_alphapose_result.json'.format(kps_alpha_format))

        self.coco2lsp = [10, 9, 8, 11, 12, 13, 4, 3, 2, 5, 5, 6, 1, 0]
        self.mpii_2_lsp14 = [0,1,2,3,4,5, 10,11,12,13,14,15, 8,9]
        self.coco25_2_lsp14 = [24, 1,2, 3,4,23, 6,7,8,9,10,11,12,13]
        set_names = ['train']#'test','train','validation'

        self.scale_range = [1.5,1.7]

        print('Start loading 3DPW data.')
        self.labels = []
        self.label_dict = {}
        self.error = []
        self.label_idx = 0
        for set_name in set_names:
            label_dir = os.path.join(self.data3d_dir,set_name)
            self.get_labels(label_dir)
        if kps_alpha_format=='mpii':
            self.load_alphapose_mpii()
        else:
            self.load_alphapose_coco()

        print('3DPW dataset total {} samples'.format(self.__len__()))

    def get_labels(self,label_dir):
        label_paths = glob.glob(label_dir+'/*.pkl')
        annots_hmrvideo = np.load(os.path.join(self.data_folder,"annots_hmrvideo.npz"),allow_pickle=True)['annots'][()]
        for label_path in label_paths:
            raw_labels = self.read_pkl(label_path)
            for i in range(len(raw_labels['poses'])):
                frame_num = len(raw_labels['img_frame_ids'])
                valid_idx = []
                for j in range(frame_num):
                    label = {}
                    label['name'] = raw_labels['sequence']
                    label['ids'] = j
                    label['frame_ids'] = raw_labels['img_frame_ids'][j]
                    label['subject_ids'] = i
                    label['kp2d'] = raw_labels['poses2d'][i][j]
                    if label['kp2d'][2].sum()<1:
                        self.error.append([label['name'],i,j])
                        continue
                    label['betas'] = annots_hmrvideo[label['name']+'.pkl']['shape'][i]
                    label['poses'] = raw_labels['poses'][i][j]
                    label['t'] = raw_labels['trans'][i][j]

                    valid_idx.append(j)
                    label['kp3d'] = annots_hmrvideo[label['name']+'.pkl']['kp3d'][i,j][self.coco25_2_lsp14]#
                    label['cam_trans'] = raw_labels['cam_poses'][j,:3,3]
                    label['cam_rotation_matrix'] = raw_labels['cam_poses'][j,:3,:3]
                    label['campose_valid_mask'] = raw_labels['campose_valid'][i][j]
                    self.labels.append(label)
                    label_dict_name = '{}_{}'.format(label['name'],label['subject_ids'])
                    if label_dict_name not in self.label_dict:
                        self.label_dict[label_dict_name] = {label['ids']:self.label_idx}
                    else:
                        self.label_dict[label_dict_name][label['ids']] = self.label_idx
                    self.label_idx += 1
        return True

    def load_alphapose_mpii(self):
        with open(self.kps_alpha_json_file,'r') as f:
            raw_labels = json.load(f)
        error_num = 0
        for idx,annot_3d in enumerate(self.labels):
            content = raw_labels['{}-image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids'])]
            poses = []
            for pid in range(len(content)):
                poses.append(np.array(content[pid]['keypoints']).reshape(-1,3)[:,:3])
            poses = np.array(poses)[:,self.mpii_2_lsp14]
            kps_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14][:-2]
            vis = np.where(kps_gt[:,2]>0)[0]
            poses_comp = poses[:,vis,:2]
            kps_gt = kps_gt[vis,:2][None,:,:]
            mis_errors = np.mean(np.linalg.norm(poses_comp-kps_gt,ord=2,axis=-1),-1)
            pose = poses[np.argmin(mis_errors)]

            pose[pose[:,2]<0.01,2] = 0
            pose[pose[:,2]>0.01,2] = 1
            annot_3d['kps_alpha'] = pose

    def load_alphapose_coco(self):
        with open(self.kps_alpha_json_file,'r') as f:
            raw_labels = json.load(f)
        frame_num = len(raw_labels)
        print('frame_num',frame_num)
        error_count=0
        for idx,annot_3d in enumerate(self.labels):
            try:
                content = raw_labels['{}-image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids'])]['bodies']
                poses = []
                for pid in range(len(content)):
                    poses.append(np.array(content[pid]['joints']).reshape(-1,3))
                poses = np.array(poses)[:,self.coco18_2_lsp14]
                poses[:,-1,2] = 0
                kps_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14][:-2]
                vis = np.where(kps_gt[:,2]>0)[0]
                mis_errors = []
                for i in range(len(poses)):
                    poses_comp = poses[i,vis]
                    vis_pred = poses_comp[:,2]>0
                    poses_comp = poses_comp[vis_pred,:2]
                    kps_gti = kps_gt[vis,:2][vis_pred,:]
                    mis_errors.append(np.mean(np.linalg.norm(poses_comp-kps_gti,ord=2,axis=-1)))
                mis_errors = np.array(mis_errors)
                pose = poses[np.argmin(mis_errors)]

                pose[pose[:,2]<0.1,2] = 0
                pose[pose[:,2]>0.1,2] = 1
                annot_3d['kps_alpha'] = pose
            except :
                print('{}/image_{:05}.jpg'.format(annot_3d['name'],annot_3d['ids']))
                error_count+=1
                pose_gt = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14]
                pose_gt[pose_gt[:,2]<0.1,2] = 0
                pose_gt[pose_gt[:,2]>0.1,2] = 1
                annot_3d['kps_alpha'] = pose_gt
        print('error_count',error_count)

    def __len__(self):
        return self.label_idx

    def get_item_video(self,index):
        label = self.labels[index]
        label_dict_name = '{}_{}'.format(label['name'],label['subject_ids'])
        ids_sequence = list(self.label_dict[label_dict_name].keys())
        current_frame = label['ids']
        current_spawn = int((self.spawn-1)/2)
        features_idx = []
        for index, num in enumerate(list(range(current_frame, current_frame+current_spawn+1))):
            if num not in ids_sequence:
                num=features_idx[index-1]
            features_idx.append(num)
        for index, num in enumerate(list(range(current_frame-1, current_frame-current_spawn-1,-1))):
            if num not in ids_sequence:
                num=features_idx[0]
            features_idx=[num]+features_idx
        labels_idx = []
        for idx in features_idx:
            labels_idx.append(self.label_dict[label_dict_name][idx])
        video = []
        video_input = {}
        for label_idx in labels_idx:
            video.append(self.get_item_single_frame(label_idx))
        for key in video[0].keys():
            if key=='image':
                video_input[key] = torch.cat([video[i][key].unsqueeze(0) for i in range(len(video))])
            elif key=='kps_alpha':
                video_input[key] = torch.cat([video[i][key].unsqueeze(0) for i in range(len(video))])
            else:
                video_input[key] = video[current_spawn][key]
        return video_input

    def get_item_single_frame(self,index):
        annot_3d = self.labels[index]
        imgpath = os.path.join(self.image_dir,annot_3d['name'],'image_{:05}.jpg'.format(annot_3d['ids']))
        name = os.path.join(self.image_dir,annot_3d['name'],'image_{:05}_{}.jpg'.format(annot_3d['ids'],annot_3d['subject_ids']))
        orgimage = cv2.imread(imgpath)[:,:,::-1].copy()

        kps_alpha = annot_3d['kps_alpha']

        kps = annot_3d['kp2d'].copy().T[self.coco18_2_lsp14]
        kps[-1,2]=0
        kp3d = annot_3d['kp3d'].copy()

        kp3d_mono = kp3d.copy()

        theta,beta,t = annot_3d['poses'].copy(),annot_3d['betas'].copy(),annot_3d['t'].copy()
        global_rotation = theta[:3].copy()
        params = np.concatenate([t,theta,beta])
        box = calc_aabb(kps[kps[:,2]>0])
        scale = self.scale_range[0]

        image, kps, kps_offset = cut_image(orgimage.copy(), kps, scale, box[0]-[0,80], box[1])
        offset,lt,rb,size,_ = kps_offset
        leftTop = [lt[0]-offset[0],lt[1]-offset[1]]
        kps_alpha = off_set_pts(kps_alpha, leftTop)
        offsets = np.array([image.shape[1],image.shape[0],lt[1],rb[1],lt[0],rb[0],offset[1],size[1],offset[0],size[0]],dtype=np.int)

        kps = self.process_kps(kps,image)
        kps_alpha = self.process_kps(kps_alpha,image)

        dst_image = cv2.resize(image, (self.crop_size, self.crop_size), interpolation = cv2.INTER_CUBIC)
        if self.high_resolution:
            org_image = cv2.resize(dst_image, (500,500), interpolation = cv2.INTER_CUBIC)
        else:
            org_image = dst_image

        return {
            'image': torch.from_numpy(convert_image_by_pixformat_normalize(dst_image, 'NCHW', True)).float(),
            'offsets': torch.from_numpy(offsets),
            'image_org': torch.from_numpy(org_image),
            'kp_2d': torch.from_numpy(kps.astype(np.float32)).float(),
            'kps_alpha': torch.from_numpy(kps_alpha.astype(np.float32)).float(),
            'kp_3d': torch.from_numpy(kp3d.astype(np.float32)).float(),
            'kp3d_mono':  torch.from_numpy(kp3d_mono.astype(np.float32)).float(),
            'param': torch.from_numpy(params.astype(np.float32)).float(),
            'global_rotation': torch.from_numpy(global_rotation.astype(np.float32)).float(),
            'imgpath': imgpath,
            'name':name,
            'data_set':'3DPW'}

if __name__ == '__main__':
    dataset=PW3D(spawn = 9,video=False)
    test_dataset(dataset,with_3d=True)
    print('Done')
