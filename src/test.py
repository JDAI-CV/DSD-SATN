
from base import *
from visualization import Visualizer
from eval import val_result

class Demo(Base):
    def __init__(self):
        super(Demo, self).__init__()
        self._build_model()
        self.loader_val = self._create_data_loader(train_flag=False)
        self.visualizer = Visualizer(high_resolution=self.high_resolution)
        print('Initialization finished!')

    def evaluation(self):
        if self.internet and not self.online:
            self.test_internet()
            return True
        elif self.test_single:
            self.test_single_frame()
            return True
        elif self.online:
            self.test_online()
            return True
        if self.eval_with_single_frame_network:
            self.test_entire_net()
        else:
            val_result(self,'eval',evaluation = True)

    def test_entire_net(self):
        MPJPE = AverageMeter()
        PA_MPJPE = AverageMeter()
        integral_kp2d_error = AverageMeter()
        PA_MPJPE_results, MPJPE_results, imgpaths = [], [], []
        with torch.no_grad():
            for test_iter,data_3d in enumerate(self.loader_val):
                imgs = data_3d['image'].cuda()
                video_frame_num, video_spawn = imgs.shape[:2]
                imgs = imgs.reshape(int(video_frame_num*video_spawn),imgs.shape[2],imgs.shape[3],imgs.shape[4])
                kps_alphas = data_3d['kps_alpha'].reshape(int(video_frame_num*video_spawn),14,2)
                process_data = {'image':imgs, 'kps_alpha':kps_alphas}
                real_3d = data_3d['kp_3d']

                blf,_,_ = self.net_forward(process_data,self.spatial_feature_extractor,only_blf=True,video=False)
                video_input = blf.reshape(video_frame_num,video_spawn,512)
                outputs = self.generator(video_input,kp3d_24=self.kp3d_24)

                ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs

                kp3d_mono = data_3d['kp3d_mono'].reshape(-1,self.kp3d_num,3).cuda()
                kps = predict_j2d

                predicts = j3d_camed.reshape(-1,self.kp3d_num,3)#.cpu().numpy()
                if self.eval_pw3d:
                    predicts_aligned = align_by_root(predicts)
                else:
                    predicts_aligned = align_by_pelvis(predicts)

                kp3d_mono = align_by_root(kp3d_mono)

                mpjpe_error = torch.sqrt(((predicts_aligned - kp3d_mono)**2).sum(-1)).mean()*1000
                MPJPE.update(mpjpe_error)

                per_verts_error = p_mpjpe(predict_j3d.detach().cpu().numpy(),real_3d.numpy().reshape(-1,self.kp3d_num,3),each_separate=True).mean(-1)*1000
                PA_MPJPE.update(np.mean(per_verts_error))

                if test_iter%100==0:
                    name = self.result_img_dir+'/{}_{:.3f}_{:.3f}_val_{}'.format(self.tab,MPJPE.avg,PA_MPJPE.avg,test_iter)
                    self.visualizer.visulize_result(outputs,kps,data_3d,name,vnum = 2,nokp=True,org_img=True)

                if test_iter%10==0:
                    print('evaluation {}/{}: {:.3f}, {:.3f}'.format(test_iter,len(self.loader_val),MPJPE.avg,PA_MPJPE.avg))

        PA_MPJPE_result = PA_MPJPE.avg
        MPJPE_result = MPJPE.avg

        print('-'*20)

        print('MPJPE: {:.3f}'.format(MPJPE_result))
        print('PA_MPJPE: {:.3f}'.format(PA_MPJPE_result))
        print('-'*20)

        return MPJPE_result, PA_MPJPE_result

    def test_internet(self):
        if self.video:
            self.generator.eval()
        print('evaluation in eval mode')

        with torch.no_grad():
            for test_iter,data_3d in enumerate(self.loader_val):
                imgs = data_3d['image'].cuda()

                if self.eval_with_single_frame_network:
                    video_frame_num, video_spawn = imgs.shape[:2]
                    imgs = imgs.reshape(int(video_frame_num*video_spawn),imgs.shape[2],imgs.shape[3],imgs.shape[4])
                    blf,_ = self.spatial_feature_extractor(imgs,only_blf=True)
                    video_input = blf.reshape(video_frame_num,video_spawn,512)
                    outputs = self.generator(video_input,kp3d_24=self.kp3d_24)
                else:
                    outputs,_,_ = self.generator(imgs)

                ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs
                name = self.result_img_dir+'/{}_val_{}'.format(self.tab,test_iter)
                self.visualizer.visulize_result(outputs,predict_j2d,data_3d,name,vnum = self.val_batch_size,nokp=True,org_img=True)

                if test_iter%10==0:
                    print('evaluation {}/{}'.format(test_iter,len(self.loader_val)))

        return True

    def test_single_frame(self):
        #self.generator.eval()
        print('evaluation in eval mode')
        data_dir = '/export/home/suny/dataset/fashion_data/'
        results = {}

        with torch.no_grad():
            for test_iter,data_3d in enumerate(self.loader_val):

                outputs,kps,details = self.net_forward(data_3d,self.generator,video=False)

                ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs
                name = data_3d['result_dir']

                img_paths = data_3d['imgpath']
                params = torch.cat([cam,pose,shape],dim=-1).detach().cpu().numpy()
                verts = predict_verts.cpu().numpy().astype(np.float16)
                j3d_camed = j3d_camed.detach().cpu().numpy().astype(np.float16)
                kps = predict_j2d
                for idx, img_path in enumerate(img_paths):
                    save_name = img_path.replace(data_dir,'')
                    results[save_name] = params[idx]#{'params':list(params[idx]),'j3d_camed':list(j3d_camed[idx])}
                    self.smpl.save_obj(verts[idx],name[idx].replace('.jpg','.obj'))

                self.visualizer.visulize_result(outputs,kps,data_3d,name,vnum = self.val_batch_size,nokp=True,org_img=True,keep_name=True)

                if test_iter%10==0:
                    print('evaluation {}/{}'.format(test_iter,len(self.loader_val)))
            f=h5py.File(os.path.join(data_dir,'Deepfashion_MEN_smpl_results.h5'),'w')#Deepfashion_MEN
            for item, value in results.items():
                f[item] = value

        print('-'*20)

        return True

    def save_features_fn(self):
        #self.save_result_single(self.loader_val)
        #self.save_features_single(self.loader_val)
        self.save_features_pa(self.loader_val)#(self.loader_val)

    def save_features_pa(self,data_loader):
        #self.generator.eval()
        annots = {}
        param = {}
        with torch.no_grad():
            for test_iter,data_3d in enumerate(data_loader):
                imgs = data_3d['image'].cuda()
                imgpaths = data_3d['imgpath']

                bilinearfeature,kps,params = self.generator(imgs,only_blf=True)
                bilinear = bilinearfeature.cpu().numpy()
                params = params.cpu().numpy()

                for idx,imgpath in enumerate(imgpaths):
                    if self.with_h36m:
                        img_name = os.path.basename(imgpath)
                    elif self.with_pa:
                        video_name,frame_name = imgpath.split('/')[-2:]
                        img_name = '{}-{}'.format(video_name,frame_name)
                    annots[img_name] = bilinear[idx]
                    param[img_name] = params[idx]
                if test_iter%1000==0:
                    print(test_iter,img_name)
        dataset = '3DPW'#'h36m' if self.with_h36m else 'Penn_Action'
        np.savez("/export/home/suny/dataset/{}/annots_test_bilinearf.npz".format(dataset),annots=annots,params=param)#

    def save_features_single(self,data_loader):
        with torch.no_grad():
            for test_iter,data_3d in enumerate(data_loader):
                imgs = data_3d['image'].cuda()
                imgpaths = data_3d['imgpath']

                outputs,kps,details,backbonefeature,bilinearfeature,params = self.generator(imgs,allfeature=True)
                for idx,imgpath in enumerate(imgpaths):
                    newpath = imgpath.rstrip('.jpg')+'_cropped_features_kp2d_best.npz'
                    np.savez(newpath, \
                    backbone = backbonefeature[idx].cpu().numpy(),
                    bilinear = bilinearfeature[idx].cpu().numpy(),
                    details = details[idx].cpu().numpy(),
                    kp = kps[idx].cpu().numpy(),
                    params=params[idx].cpu().numpy())
                if test_iter%1000==0:
                    print(test_iter,newpath)
    def save_result_single(self,data_loader):
        with torch.no_grad():
            for test_iter,data_3d in enumerate(data_loader):
                imgs = data_3d['image'].cuda()
                imgpaths = data_3d['imgpath']

                outputs,kps,details,backbonefeature,bilinearfeature,params = self.generator(imgs,allfeature=True)
                for idx,imgpath in enumerate(imgpaths):
                    #newpath = './dataset/test/'+os.path.basename(imgpath.rstrip('.jpg')+'_videoresult.npy')
                    newpath = imgpath.rstrip('.jpg')+'_videoresult.npy'
                    np.save(newpath, params[idx].cpu().numpy())
                if test_iter%1000==0:
                    print(test_iter,newpath)

    def test_online(self,fast_mode=False,webcam=False,bs=2):
        os.chdir('./AP')
        show_size = 512

        data_loader = WebcamLoader(0,batchsize=bs).start() if webcam else VideoLoader(self.videofile, batchSize=bs).start()
        (fourcc,fps,frameSize) = data_loader.videoinfo()

        # Load detection loader
        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader_web(data_loader, batchSize=bs).start() if webcam else DetectionLoader(data_loader, batchSize=bs).start()
        det_processor = DetectionProcessor_fast(det_loader, batchsize=bs).start()
        ovisualizer = Visualizer_online(save_video=True, vis=True,savepath=self.save_path_video, fps=fps, frameSize=(frameSize[0]+show_size,frameSize[1]),show_size=show_size).start()

        runtime_profile = {'dt': [],'pt': [],'pn': []}

        count = 0
        batchSize = bs*2
        while True:
            start_time = getTime()
            with torch.no_grad():
                while det_processor.len()==0:
                    time.sleep(0.001)
                inpss, org_imgs, orig_imgs, kps_offsets, frame_nums,boxess= [], [], [], [], [],[]
                while det_processor.len()>0:
                    output = det_processor.read()
                    (inps,org_img, orig_img, im_name, boxes, scores, kps_offset,frame_num) = output

                    if orig_img is None or boxes is None:
                        continue

                    inpss.append(inps)
                    org_imgs.append(org_img[:,:,::-1])
                    orig_imgs.append(orig_img)
                    kps_offsets.append(kps_offset)
                    frame_nums.append(frame_num)
                    boxess.append(boxes)
                    if len(inpss)==8:
                        break
                inps, org_imgs, orig_imgs, kps_offsets = torch.cat(inpss), np.stack(org_imgs), np.stack(orig_imgs), np.stack(kps_offsets)

                datalen = inps.size(0)
                print(count,'Number of images: ',datalen)

                ckpt_time, det_time = getTime(start_time)
                runtime_profile['dt'].append(det_time)

                if batchSize > 1:
                    leftover = 0
                    if (datalen) % batchSize:
                        leftover = 1
                    num_batches = datalen // batchSize + leftover
                    vertex = []
                    for j in range(num_batches):
                        imgs = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                        outputs,_,_ = self.generator(imgs)
                        vertex.append(outputs[5])
                    verts_camed = torch.cat(vertex)
                else:
                    imgs = inps.cuda()
                    outputs,_,_ = self.generator(imgs)
                    verts_camed = outputs[5]

                ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs

                ckpt_time, pose_time = getTime(ckpt_time)
                runtime_profile['pt'].append(pose_time)

                name = './test_online2/{}'.format(count)
                ovisualizer.add(predict_verts,org_imgs,orig_imgs,frame_nums,boxess,name)
                #self.visulize_result_fast(verts_camed,org_imgs,orig_imgs,kps_offsets,name)

                ckpt_time, post_time = getTime(ckpt_time)
                runtime_profile['pn'].append(post_time)

                print('det time: {dt:.3f} | estimation time: {pt:.2f} | rendering: {pn:.4f}'.format(
                    dt=runtime_profile['dt'][-1], pt=runtime_profile['pt'][-1], pn=runtime_profile['pn'][-1]))

                count+=1

        return True

    def visulize_result_fast(self,verts_camed,imgs,org_imgs,offsets,name, white_background=False):

        if white_background:
            rendered_imgs = self.visualize_renderer(verts_camed, np.ones_like(imgs)*255)
        else:
            for i in range(len(verts_camed)):
                self.plot_3dmesh_fast(verts_camed[i].detach().cpu().numpy(), name+'_{}.jpg'.format(i))

    def plot_3dmesh_fast(self,verts,name):
        plt = plot_mesh(verts, self.smpl.faces, subplot = [1,1,1], title = 'mesh', el = 90, az = -90, lwdt=.1, dist = 6, color = "orange")
        plt.savefig(name)

if args.online:
    import sys
    sys.path.append('./AP')

    from dataloader import WebcamLoader,VideoLoader, DetectionLoader_web,DetectionLoader, DetectionProcessor_fast, DataWriter, Mscoco
    print('pass import')
    from yolo.util import write_results, dynamic_write_results
    print('pass import')
    from SPPE.src.main_fast_inference import *
    print('pass import')

    import ntpath
    from fn import getTime

    from pPose_nms import pose_nms, write_json
    import torch.multiprocessing as multiprocessing
    from utils.online_visualizer import Visualizer_online
    print('pass import')

def main():
    demo = Demo()
    if args.save_features:
        demo.save_features_fn()
        return True
    elif not args.eval:
        demo.train()
    elif args.eval:
        pass
    else:
        copy_state_dict(demo.generator.state_dict(),torch.load(demo.best_save_path),prefix = 'module.')
    demo.evaluation()

if __name__ == '__main__':
    main()