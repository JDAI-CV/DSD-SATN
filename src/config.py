import os
import argparse
import math

code_dir = os.path.abspath(__file__).replace('config.py','')
project_dir = os.path.abspath(__file__).replace('src/config.py','')

#setting PATH of each dataset
data_set_path = {
    'h36m':'/media/sunyu/sunyu1/dataset/h36m',
    'aich':'/media/sunyu/sunyu1/dataset/ai_challenger',
    'mpii':'/media/sunyu/sunyu1/dataset/mpii',
    'up':'/media/sunyu/sunyu1/dataset/UP',
    'pw3d':'/media/sunyu/sunyu1/dataset/3DPW',
    'mosh':'/media/sunyu/sunyu1/dataset/mosh_gen',}

parser = argparse.ArgumentParser(description = 'SATN')
parser.add_argument('--tab',type = str,default = '',help = 'additional tabs')

mode_group = parser.add_argument_group(title='mode options')
#mode setting
mode_group.add_argument('--eval',action='store_true',help = 'whether to evaluation')
mode_group.add_argument('--eval-pw3d',action='store_true',help = 'whether to evaluation on 3DPW dataset')
mode_group.add_argument('--online',action='store_true',help = 'whether to run online')
mode_group.add_argument('--videofile', type=str, default='./demo/demo.mp4',help='path of input video')
mode_group.add_argument('--save_path_video', type=str, default='./result.avi',help='path of saving video result')
mode_group.add_argument('--internet',action='store_true',help = 'whether to run on internet images/videos')
mode_group.add_argument('--test-single',action='store_true',help = 'whether to run on single image mode')
mode_group.add_argument('--inimg_dir',type = str,default = os.path.join(project_dir,'resources/image'),help = 'path of images for input')
mode_group.add_argument('--visual-all',action='store_true',help = 'whether to visualize all rendering results.')
mode_group.add_argument('--save-smpl-params',action='store_true',help = 'whether to save smpl parameters of 3D body mesh.')
mode_group.add_argument('--save-obj',action='store_true',help = 'whether to save obj file of 3D body mesh.')

model_group = parser.add_argument_group(title='model settings')
#model settings
model_group.add_argument('--with-kps',action='store_true',help = 'whether to predict with kps results from 2D pose estimator e.g. alphapose')
model_group.add_argument('--kps-type',type = str,default = 'kps_alpha',help = 'use ground truth or alphapose predictions: kp_2d, kps_alpha')
model_group.add_argument('--alpha-format',type = str,default = 'mpii',help = 'format of input 2D keypoints of alphapose: coco, mpii')
model_group.add_argument('--feature-num-deconv',default = 256,type = int, help = 'feature map number of deconvolutional layer of heatmap.')
model_group.add_argument('--num-backbone-layers', type = int, default = 50, help = 'choose backbone layers, 50, 101, 152')
model_group.add_argument('--gmodel-path',type = str,default = '',help = 'trained model path of generator')
model_group.add_argument('--dmodel-path',type = str,default = '',help = 'trained model path of discriminator')
model_group.add_argument('--best-save-path',type = str,default = '',help = 'trained model path of best generator')

tmodel_group = parser.add_argument_group(title='temporal model settings')
#video model settings
tmodel_group.add_argument('--video',action='store_true',help = 'whether is video')
tmodel_group.add_argument('--shuffle-aug',action='store_true',help = 'whether to shuffle augmentation')
tmodel_group.add_argument('--receptive-field',type = int,default = 9,help = '243, 81, 27, 9, 3')
tmodel_group.add_argument('--self-attention',action='store_false',help = 'whether to train with self-attension')
tmodel_group.add_argument('--shuffle-smooth-loss',type=bool,default=True,help = 'whether to use randomsmooth')
tmodel_group.add_argument('--attention-layer',type = int,default = 2,help = '2,4,6')

train_group = parser.add_argument_group(title='training options')
#basic training setting
train_group.add_argument('--epoch', type = int, default = 40, help = 'training epochs')
train_group.add_argument('--fine-tune',action='store_true',help = 'whether to run online')
train_group.add_argument('--fit-epoch', type = int, default = 0, help = 'epochs for fitting model from pre-trained')
train_group.add_argument('--val-iter',type = int,default = 2000,help = 'val iter')
train_group.add_argument('--lr', help='lr',default=1e-4,type=float)
train_group.add_argument('--gpu',default=0,help='gpus',type=str)
train_group.add_argument('--batch_size',default=16,help='batch size',type=int)
train_group.add_argument('--val_batch_size',default=16,help='batch size of evaluation',type=int)
train_group.add_argument('--nw',default=6,help='number of workers',type=int)
train_group.add_argument('--features-type',type = int,default = 1,help = '{1:[512,bilinear],2:[2048,backbone]}')
train_group.add_argument('--pix-format',type = str,default = 'NCHW',help = 'format of input images')
train_group.add_argument('--use_flip',type = bool,default = False,help = 'whether to flip images')
train_group.add_argument('--normalize',type = bool,default = True,help = 'whether to normalize')
train_group.add_argument('--video-clips-input',type = bool,default = False,help = 'whether to input video clips')
train_group.add_argument('--use-cropped-img',type = bool,default = False,help = 'whether to take crppoed images as input for faster data loading')
train_group.add_argument('--adjust-lr-factor',type = float,default = 0.1,help = 'factor for adjusting the lr')
train_group.add_argument('--with-mpjpeloss',type = bool,default = True,help = 'whether to use mpjpe measure loss')
train_group.add_argument('--with-constloss',type = bool,default = False,help = 'whether to use const between kp2d and regre to measure loss')
train_group.add_argument('--best',type = float,default = 48.0,help = 'bottom level of pampjpe result for saving model')
train_group.add_argument('--best-mpjpe',type = float,default = 68.0,help = 'bottom level of mpjpe result for saving model')
train_group.add_argument('--best-pw3d',type = float,default = 48.0,help = 'bottom level of pampjpe result for saving model')
train_group.add_argument('--best-mpjpe-pw3d',type = float,default = 68.0,help = 'bottom level of mpjpe result for saving model')

dataset_group = parser.add_argument_group(title='datasets options')
#dataset setting:
dataset_group.add_argument('--dataset-rootdir',type=str, default=os.path.join(project_dir,"../../dataset/"), help= 'root dir of all datasets')
dataset_group.add_argument('--dataset',type=str, default='h36m,up,pa' ,help = 'whether use each dataset')
datasets = ['h36m','up', 'mpii', 'aich', 'mpv', 'iv', 'pa', 'df']
for dataset in datasets:
    dataset_group.add_argument('--with-{}'.format(dataset),type = bool,default = False,help = 'whether use {} dataset'.format(dataset))
dataset_group.add_argument('--use-all-subject',type = bool,default = False,help = 'whether use all subject in Human3.6M dataset')

eval_group = parser.add_argument_group(title='evaluation options')
#basic eval settings
eval_group.add_argument('--eval-protocol',type = int,default = 1,help = '1 , 2 , 3, 4')
eval_group.add_argument('--kp3d-num',type = int,default = 14,help = 'the number of kp3d')
eval_group.add_argument('--kp3d-24',type = bool,default = False,help = 'whether to use kp3d of 24')
eval_group.add_argument('--eval-with-single-frame-network',action='store_true',help = 'eval_with_single_frame_network')

other_group = parser.add_argument_group(title='other options')
#visulaization settings
other_group.add_argument('--high_resolution',type = bool,default = False,help = 'whether to visulize with high resolution 500*500')

#other settings
other_group.add_argument('--s',default=1,help='subject number Sn',type=int)
other_group.add_argument('--save-features',action='store_true',help = 'whether to just save feautures')
other_group.add_argument('--get-features',type=bool,default=False,help = 'whether to use pre-get feautures')

#model save path and log file
other_group.add_argument('--save-best-folder', type = str, default = os.path.join(project_dir,'trained_model/'), help = 'Path to save models')
other_group.add_argument('--log-path', type = str, default = os.path.join(code_dir,'log/'), help = 'Path to save log file')

smpl_group = parser.add_argument_group(title='SMPL options')
#smpl info
smpl_group.add_argument('--hmr-video-mode',type = bool,default = False,
    help = 'whether use label generated by hmr-video')
smpl_group.add_argument('--coco25-regressor-path',type = str,default = os.path.join(project_dir,"model/neutral_smpl_with_cocoplustoesankles_reg.pkl"),
    help = 'the path for coco25-regressor')
smpl_group.add_argument('--total-param-count',type = int,default = 85, help = 'the count of param param')
smpl_group.add_argument('--smpl-mean-param-path',type = str,default = os.path.join(project_dir,'model/neutral_smpl_mean_params.h5'),
    help = 'the path for mean smpl param value')
smpl_group.add_argument('--smpl-model',type = str,default = os.path.join(project_dir,'model/neutral_smpl_with_cocoplus_reg.txt'),
    help = 'smpl model path')

for name, path in data_set_path.items():
    dataset_group.add_argument('--{}-path'.format(name), default=path ,type = str, help = 'path of dataset {}'.format(name))

#loss weight setting
loss_name = ['regre','pose','shape', 'mpjpe', 'pampjpe', 'kp', 'const', 'disc', 'shuffle']
loss_weight = [10.0,  20.0,  0.6,      60.0,    80.0,     10.0,   0,     0.1,     0 ]

for name, weight in zip(loss_name, loss_weight):
    train_group.add_argument('--{}-weight'.format(name), default=weight ,type = float, help = 'weight of {} loss'.format(name))


args = parser.parse_args()

# convert some args.
args.filter_widths = [3 for i in range(int(math.log(args.receptive_field,3)))]
features_types = {1:[512,'bilinear_features'],2:[2048,'backbone_features'],3:[512,'bilinear_kp3d_features'],4:[85,'params_kp3d_features'],\
        5:[512,'bilinear_kp2d_features'],6:[85,'params_kp2d_features'],7:[512,'bilinear_kp2d_features_best'],8:[85,'params_kp2d_features_best']}
args.features_channels, args.feature_name = features_types[args.features_type]

#3:#every 64th frame of Subject 11, PA-MPJPE 4:every 64th frame of subjects (S9, S11).MPJPE
protocols = {1:'all', 2:'frontal_only', 3:'kp3d_protocol1', 4:'kp3d_protocol2'}
args.eval_mode = protocols[args.eval_protocol]

for dataset in datasets:
    exec("args.with_{}='{}' in args.dataset".format(dataset,dataset))

if args.save_features:
    args.use_all_subject=True

if args.fine_tune:
    args.epoch=10
    args.lr = 1e-4

if (args.video and args.eval_pw3d) or (args.video and args.internet):
    args.eval_with_single_frame_network = True
    args.high_resolution = True

if args.with_kps:
    args.best = 42.0
    args.best_mpjpe = 62.0

best_model_dict = {
    'dsd_single': os.path.join(project_dir,'trained_model/single_frame_dsd_model.pkl'),
    'video':os.path.join(project_dir,'trained_model/video_satn_model.pkl'),}

gmodel_path_loading = best_model_dict['dsd_single' if not args.video else 'video']
args.gmodel_path = gmodel_path_loading

dmodel_path_loading = gmodel_path_loading.replace("trained_model/","trained_model/D_")
args.dmodel_path = dmodel_path_loading

args.best_save_path = args.gmodel_path

args.tab = '{}_{}_g{}'.format(args.tab,args.dataset,args.gpu)

print('-'*16)
print('Configuration:')
print(vars(args))
print('-'*16)
print('with_kps:', args.with_kps, 'type:', args.kps_type, 'format:', args.alpha_format)
print('datasets:', args.dataset)
print('-'*16)

h36m_action_names = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',\
            'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo',\
            'Waiting', 'Walking', 'WalkDog', 'WalkTogether' ]
j14_names = ['Rankle','Rknee','Rhip','Lhip','Lknee','Lankle',\
        'Rwrist','Ralbow','Rshoulder','Lshoulder','Lalbow','Lwrist', 'neck','head']