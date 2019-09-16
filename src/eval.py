
from base import *

def val_result(self,epoch,evaluation = False,data_loader=None):
    if data_loader is None:
        data_loader = self.loader_val
    if self.video and not self.eval_pw3d:
        self.generator.eval()
    print('evaluation ...')

    MPJPE = AverageMeter()
    PA_MPJPE = AverageMeter()
    integral_kp2d_error = AverageMeter()
    integral_kp2d_pckh = AverageMeter()
    PA_MPJPE_results, MPJPE_results, imgpaths = [], [], []
    entire_length = len(data_loader)
    vnum = self.val_batch_size if self.visual_all else 6

    with torch.no_grad():
        for test_iter,data_3d in enumerate(data_loader):

            outputs,kps,details = self.net_forward(data_3d,self.generator,video=self.video)

            kps_gt = data_3d['kp_2d'].cuda().reshape(data_3d['kp_2d'].shape[0],14,2)
            vis = (kps_gt!=-2.).float()

            if not self.eval_pw3d:
                integral_kp2d_error.update((((kps_gt-kps.reshape(kps_gt.shape[0],14,-1))**2)*vis).sum(-1).mean(0).detach().cpu().numpy())
                integral_kp2d_pckh.update(compute_pckh_lsp(kps_gt.cpu().numpy(),kps.reshape(kps_gt.shape[0],14,-1).detach().cpu().numpy(),vis.cpu()))

            if test_iter%self.val_batch_size==0:# and not evaluation:
                name = self.result_img_dir+'/{}_{}_val_{}'.format(self.tab,epoch,test_iter)
                self.visualizer.visulize_result(outputs,kps,data_3d,name,vnum = vnum)
                print('PCKh: {:.3f}'.format(integral_kp2d_pckh.avg))

            ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs
 
            if self.save_smpl_params or self.save_obj:
                img_paths = data_3d['imgpath']
                verts = predict_verts.cpu().numpy().astype(np.float16)
                params = torch.cat([cam,pose,shape],dim=-1).detach().cpu().numpy()
                for idx, img_path in enumerate(img_paths):
                    save_name = os.path.join(self.result_img_dir,os.path.basename(img_path))
                    if self.save_smpl_params:
                        np.save(save_name+'.npy',params[idx])
                    if self.save_obj:
                        self.smpl.save_obj(verts[idx],save_name+'.obj')

            if self.eval_pw3d:
                h36m_idx = np.arange(len(data_3d['imgpath']))
            else:
                h36m_idx = np.where(np.array(data_3d['data_set'])=='h36m')[0]
            if len(h36m_idx)<2:
                continue

            predict_j3d = predict_j3d[h36m_idx]
            real_3d = data_3d['kp_3d'][h36m_idx]
            smpl_trans = data_3d['param'][h36m_idx, :3].cuda()
            global_rotation = data_3d['global_rotation'][h36m_idx].cuda()
            imgpaths.append(np.array(data_3d['imgpath'])[h36m_idx])

            kp3d_mono = data_3d['kp3d_mono'][h36m_idx].reshape(-1,self.kp3d_num,3).cuda()
            predicts = j3d_camed[h36m_idx].reshape(-1,self.kp3d_num,3)#.cpu().numpy()
            if self.eval_pw3d:
                predicts_aligned = align_by_pelvis(predicts)
                kp3d_mono = align_by_pelvis(kp3d_mono)
            else:
                predicts_aligned = align_by_pelvis(predicts)

            mpjpe_each = torch.sqrt(((predicts_aligned - kp3d_mono)**2).sum(-1)).mean(-1)*1000
            MPJPE_results.append(mpjpe_each.cpu())
            mpjpe_error = mpjpe_each.mean()
            MPJPE.update(mpjpe_error)

            per_verts_error = p_mpjpe(predict_j3d.detach().cpu().numpy(),real_3d.numpy().reshape(-1,self.kp3d_num,3),each_separate=True).mean(-1)*1000
            PA_MPJPE_results.append(per_verts_error)
            PA_MPJPE.update(np.mean(per_verts_error))

            if test_iter>3*self.val_batch_size and not evaluation:
                break

            if test_iter%self.val_batch_size==0 and evaluation:
                print('evaluation {}/{}: {:.3f}, {:.3f}'.format(test_iter,len(data_loader),MPJPE.avg,PA_MPJPE.avg))

    PA_MPJPE_result = PA_MPJPE.avg
    MPJPE_result = MPJPE.avg
    PA_MPJPE_acts = self.h36m_evaluation_act_wise(np.concatenate(PA_MPJPE_results,axis=0),np.concatenate(np.array(imgpaths),axis=0))
    MPJPE_acts = self.h36m_evaluation_act_wise(np.concatenate(MPJPE_results,axis=0),np.concatenate(np.array(imgpaths),axis=0))

    print('-'*20)
    print('MPJPE: {:.3f}'.format(MPJPE_result))
    print('PA_MPJPE: {:.3f}'.format(PA_MPJPE_result))
    print('-'*20)
    table = PrettyTable(['Protocol']+config.h36m_action_names)
    table.add_row(['1']+MPJPE_acts)
    table.add_row(['2']+PA_MPJPE_acts)
    print(table)
    print('-'*20)
    print('integral_kp2d_PCKh:',integral_kp2d_pckh.avg)
    print('-'*20)
    if not self.eval_pw3d:
        scale_factor = 256
        print('integral_kp2d_error:')
        table = PrettyTable(['部位']+config.j14_names+['均值'])
        table.add_row(['pixel error']+np.array(integral_kp2d_error.avg*scale_factor,dtype=np.float16).astype(np.str).tolist()\
            +['{:.2f}'.format(integral_kp2d_error.avg.mean()*scale_factor)])
        print(table)
        print('-'*20)
    if evaluation:
        print(self.gmodel_path)
        print(self.best_save_path)
    return MPJPE_result, PA_MPJPE_result
