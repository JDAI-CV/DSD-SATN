import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from dataset.image_base import *

class Demo_Loader(Image_base):
    def __init__(self,train_flag =False,high_resolution = False):
        super(Demo_Loader,self).__init__(train_flag,high_resolution)
        self.train_flag=False
        self.scale_range = [1.0,1.0]
        self.image_dir = args.inimg_dir
        self.kps_dir = args.inimg_dir
        self.image_save_dir = os.path.join(args.inimg_dir,'../results')
        check_and_mkdir(self.image_save_dir)
        self.augment_half = False
        self.empty_kp2d = np.ones((14,3))*-2

        self.images = os.listdir(self.image_dir)
        print('Demo dataset total {} samples'.format(self.__len__()))

    def __len__(self):
        return len(self.images)

    def get_image_info(self,index):
        img_file_path = self.images[index]

        imgpath = os.path.join(self.image_dir,img_file_path)
        frame = cv2.imread(imgpath)[:,:,::-1]
        result_dir = os.path.join(self.image_save_dir,os.path.basename(img_file_path))

        box = np.array([[0,0],[frame.shape[1],frame.shape[0]]])

        metas = ('Demo',imgpath,result_dir,self.empty_kp3d,self.empty_kp3d,self.empty_param,self.empty_gr)

        return frame, self.empty_kp2d, box, metas


if __name__ == '__main__':
    dataset=Demo_Loader()
    test_dataset(dataset)
    print('Done')