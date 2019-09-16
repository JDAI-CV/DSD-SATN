import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from dataset.image_base import *

class UP(Image_base):
    def __init__(self,train_flag = True,high_resolution = False):
        super(UP,self).__init__(train_flag,high_resolution)
        self.data_folder = config.data_set_path['up']#os.path.join(self.data_folder,'UP/')
        self.data3d_dir = os.path.join(self.data_folder,'up-3d')
        self.segmentation_dir = os.path.join(self.data_folder,'segmentation')
        self.save_dir = os.path.join(self.data_folder,'result/')
        self.augment_half = True

        self.scale_dir = os.path.join(self.data_folder,'p14_joints/scale_14_500_p14_joints.txt')
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [6, 11], [8, 9], [7, 10]]
        self.scale_range = [1.3,1.5]
        self.rotation_factor = 30

        print('Start loading up data.')
        self.high_qulity_idx = self.get_high_qulity_idx()
        print('UP dataset total {} samples'.format(len(self.high_qulity_idx)))

    def get_high_qulity_idx(self):
        files = glob.glob(os.path.join(self.data3d_dir,'*_quality_info.txt'))
        high_qulity_idx = []
        for file in files:
            quality = self.read_txt(file)
            data_idx = os.path.basename(file).split('_')[0]
            dataset_info_dir = os.path.join(self.data3d_dir,'{}_dataset_info.txt'.format(data_idx))
            dataset_info = self.read_txt(dataset_info_dir)[0]
            if 'high\n' in quality and dataset_info!='fashionpose':
                high_qulity_idx.append(data_idx)
        return high_qulity_idx

    def read_txt(self,file_path):
        f=open(file_path)
        lines = f.readlines()
        if len(lines)!=1:
            print('different crop_fit_info lines of {}:'.format(file_path), len(lines))
        info = lines[0].split(' ')
        return info

    def __len__(self):
        return len(self.high_qulity_idx)

    def get_image_info(self,index):
        index = self.high_qulity_idx[index]
        imgpath = os.path.join(self.data3d_dir,'{}_image.png'.format(index))
        image = cv2.imread(imgpath)[:,:,::-1]

        annot_3d_dir = os.path.join(self.data3d_dir,'{}_body.pkl'.format(index))
        annot_3d = self.read_pkl(annot_3d_dir)

        theta,beta,t = annot_3d['pose'],annot_3d['betas'],annot_3d['t']
        params = torch.from_numpy(np.concatenate([t,theta,beta])).float()

        annot_2d_kp_dir = os.path.join(self.data3d_dir,'{}_joints.npy'.format(index))
        kp2d = self.read_npy(annot_2d_kp_dir).T

        box = calc_aabb(kp2d[kp2d[:,2]>0])
        result_dir = '{}/{}'.format(self.save_dir,os.path.basename(imgpath))

        metas = ('UP',imgpath,result_dir,self.empty_kp3d,self.empty_kp3d,params,self.empty_gr)

        return image, kp2d, box, metas


if __name__ == '__main__':
    dataset=UP()
    test_dataset(dataset,with_3d=True)
    print('Done')