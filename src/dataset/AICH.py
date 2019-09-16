import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from dataset.image_base import *

class AICH(Image_base):
    def __init__(self,train_flag=True,high_resolution=False):
        super(AICH,self).__init__(train_flag,high_resolution)
        self.data_folder = config.data_set_path['aich']
        self.scale_range = [1.3, 1.4]
        self.max_intersec_ratio = 0.9
        self.only_single_person = True
        self.min_pts_required = 5
        folder_dir = "ai_challenger_keypoint_train_20170909/" if self.train_flag else "ai_challenger_keypoint_validation_20170911/"
        self.data_folder = os.path.join(self.data_folder, folder_dir)

        self.img_ext = '.jpg'
        self.save_dir = os.path.join(self.data_folder,"result/")
        self._load_data_set()

    def _load_data_set(self):
        self.images, self.kp2ds, self.boxs = [], [], []
        print('start loading AI CH keypoint data.')
        anno_file = 'keypoint_train_annotations_20170909.json' if self.train_flag else "keypoint_validation_annotations_20170911.json"

        anno_file_path = os.path.join(self.data_folder, anno_file)
        with open(anno_file_path, 'r') as reader:
            anno = json.load(reader)
        for record in anno:
            image_name = record['image_id'] + self.img_ext
            image_folder = 'keypoint_train_images_20170902' if self.train_flag else "keypoint_validation_images_20170911"
            image_path = os.path.join(self.data_folder, image_folder, image_name)
            kp_set = record['keypoint_annotations']
            box_set = record['human_annotations']
            self._handle_image(image_path, kp_set, box_set)

        print('finished load Ai CH keypoint data, total {} samples'.format(len(self)))

    def _ai_ch_to_lsp(self, pts):
        kp_map = [8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 13, 12]
        pts = np.array(pts, dtype = np.float).reshape(14, 3).copy()
        pts[:, 2] = (3.0 - pts[:, 2]) / 2.0
        return pts[kp_map].copy()

    def _handle_image(self, image_path, kp_set, box_set):
        assert len(kp_set) == len(box_set)
        if len(kp_set) > 1:
            if self.only_single_person:
                return
        for key in kp_set.keys():
            kps = kp_set[key]
            box = box_set[key]
            self._handle_sample(key, image_path, kps, [ [box[0], box[1]], [box[2], box[3]] ], box_set)

    def _handle_sample(self, key, image_path, pts, box, boxs):
        def _collect_box(key, boxs):
            r = []
            for k, v in boxs.items():
                if k == key:
                    continue
                r.append([[v[0],v[1]], [v[2],v[3]]])
            return r

        def _collide_heavily(box, boxs):
            for it in boxs:
                if get_rectangle_intersect_ratio(box[0], box[1], it[0], it[1]) > self.max_intersec_ratio:
                    return True
            return False
        pts = self._ai_ch_to_lsp(pts)
        valid_pt_cound = np.sum(pts[:, 2])
        if valid_pt_cound < self.min_pts_required:
            return

        self.images.append(image_path)
        self.kp2ds.append(pts)

    def __len__(self):
        return len(self.images)

    def get_image_info(self,index):
        imgpath = self.images[index]
        kp2d = self.kp2ds[index].copy()
        box = calc_aabb(kp2d[kp2d[:,2]>0])
        image = cv2.imread(imgpath)[:,:,::-1]
        result_dir = '{}/{}'.format(self.save_dir,os.path.basename(imgpath))
        metas = ('AICH',imgpath,result_dir,self.empty_kp3d,self.empty_kp3d,self.empty_param,self.empty_gr)
        return image, kp2d, box, metas

if __name__ == '__main__':
    dataset = AICH(train_flag=True)
    test_dataset(dataset)
    print('Done')
