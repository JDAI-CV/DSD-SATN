import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('dataset/',''))
from dataset.image_base import *
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints

class MPIIDataset(Image_base):
    def __init__(self, train_flag=True,high_resolution = False):
        super(MPIIDataset,self).__init__(train_flag,high_resolution)
        self.data_folder = config.data_set_path['mpii']
        self.num_joints = 16
        self.scale_factor = 0.2
        self.rotation_factor = 20
        self.const_box = [np.array([0,0]),np.array([256,256])]

        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.parent_ids = [1, 2, 6, 6, 3, 4, 6, 6, 7, 8, 11, 12, 7, 7, 13, 14]
        self.save_dir = os.path.join(self.data_folder,'result/')

        if self.train_flag:
        	self.image_set='train'
        else:
        	self.image_set='valid'

        print('Start loading MPII data.')
        self._get_db()
        print('Loaded total {} samples'.format(len(self.gt_db)))

    def _get_db(self):
        # create train/val split
        file_name = os.path.join(self.data_folder,'annot',self.image_set+'.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        self.gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float32)
            s = np.array([a['scale'], a['scale']], dtype=np.float32)

            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25
            c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float32)
            joints_3d_vis = np.zeros((self.num_joints,  3), dtype=np.float32)
            joints = np.array(a['joints'])
            joints[:, 0:2] = joints[:, 0:2] - 1
            joints_vis = np.array(a['joints_vis'])
            assert len(joints) == self.num_joints, \
                'joint num diff: {} vs {}'.format(len(joints),
                                                  self.num_joints)

            joints_3d[:, 0:2] = joints[:, 0:2]
            joints_3d_vis[:, 0] = joints_vis[:]
            joints_3d_vis[:, 1] = joints_vis[:]

            self.gt_db.append({
                'image': os.path.join(self.data_folder, 'images', image_name),
                'center': c,
                'scale': s,
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                })

    def __len__(self):
        return len(self.gt_db)

    def get_image_info(self,index):

        info = self.gt_db[index]
        imgpath = info['image']
        image = cv2.imread(imgpath)[:,:,::-1]
        joints = info['joints_3d']
        joints_vis = info['joints_3d_vis'][:, 0]

        c = info['center']
        s = info['scale']
        r = 0
        if self.train_flag:
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

        trans = get_affine_transform(c, s, r, (self.crop_size, self.crop_size))
        dst_image = cv2.warpAffine(image,trans,
            (self.crop_size, self.crop_size),flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_vis[i] > 0.0:
                joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)
        kp2d = np.concatenate([joints[:,0:2],joints_vis[:,None]],1)[self.mpii_2_lsp14]

        result_dir = '{}/{}'.format(self.save_dir,os.path.basename(imgpath))
        metas = ('mpii',imgpath,result_dir,self.empty_kp3d,self.empty_kp3d,self.empty_param,self.empty_gr)

        return dst_image, kp2d, self.const_box, metas

    def evaluate(self, preds, output_dir=None, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            scio.savemat(pred_file, mdict={'preds': preds})

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.data_folder,
                               'annot',
                               'gt_{}.mat'.format('valid'))
        gt_dict = scio.loadmat(gt_file)
        dataset_joints = gt_dict['dataset_joints']
        jnt_missing = gt_dict['jnt_missing']
        pos_gt_src = gt_dict['pos_gt_src']
        headboxes_src = gt_dict['headboxes_src']

        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(dataset_joints == 'head')[1][0]
        lsho = np.where(dataset_joints == 'lsho')[1][0]
        lelb = np.where(dataset_joints == 'lelb')[1][0]
        lwri = np.where(dataset_joints == 'lwri')[1][0]
        lhip = np.where(dataset_joints == 'lhip')[1][0]
        lkne = np.where(dataset_joints == 'lkne')[1][0]
        lank = np.where(dataset_joints == 'lank')[1][0]

        rsho = np.where(dataset_joints == 'rsho')[1][0]
        relb = np.where(dataset_joints == 'relb')[1][0]
        rwri = np.where(dataset_joints == 'rwri')[1][0]
        rkne = np.where(dataset_joints == 'rkne')[1][0]
        rank = np.where(dataset_joints == 'rank')[1][0]
        rhip = np.where(dataset_joints == 'rhip')[1][0]

        jnt_visible = 1 - jnt_missing
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']

if __name__ == '__main__':
    dataset = MPIIDataset(train_flag=True)
    test_dataset(dataset)
    print('Done')
