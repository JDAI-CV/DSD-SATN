
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
        if self.test_single:
            self.test_single_frame()
        elif self.eval_with_single_frame_network:
            self.test_entire_net()
        else:
            val_result(self,'eval',evaluation = True)

    def test_single_frame(self):
        results = {}
        with torch.no_grad():
            for test_iter,data_3d in enumerate(self.loader_val):
                outputs,kps,details = self.net_forward(data_3d,self.generator,video=False)
                ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs
                names = data_3d['result_dir']
                img_paths = data_3d['imgpath']
                params = torch.cat([cam,pose,shape],dim=-1).detach().cpu().numpy()
                verts = predict_verts.cpu().numpy().astype(np.float16)
                j3d_camed = j3d_camed.detach().cpu().numpy().astype(np.float16)
                for idx, save_name in enumerate(names):
                    #save_name = os.path.join(self.result_img_dir,os.path.basename(img_path))
                    results[save_name] = params[idx]
                    self.smpl.save_obj(verts[idx],save_name+'.obj')

                self.visualizer.visulize_result(outputs,kps,data_3d,names,vnum = self.val_batch_size,nokp=True,org_img=True,keep_name=True)
                if test_iter%10==0:
                    print('evaluation {}/{}'.format(test_iter,len(self.loader_val)))
            f=h5py.File(os.path.join(self.result_img_dir,'SMPL_results.h5'),'w')
            for item, value in results.items():
                f[item] = value

        return True

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

                predicts = j3d_camed.reshape(-1,self.kp3d_num,3)
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


def main():
    demo = Demo()
    demo.evaluation()

if __name__ == '__main__':
    main()