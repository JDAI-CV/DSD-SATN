from base import *
import utils.neuralrenderer_render as nr

class Visualizer(object):
    def __init__(self,high_resolution=False):
        self.high_resolution = high_resolution
        self.renderer = nr.get_renderer(high_resolution=self.high_resolution).cuda()

    def visualize_renderer(self,verts,images):
        #verts = torch.from_numpy(verts).cuda()
        #verts = self.batch_orth_proj_verts(verts,cam)
        #verts = torch.cat((verts[:,:,1].unsqueeze(-1),\
        #    -verts[:,:,2].unsqueeze(-1),verts[:,:,0].unsqueeze(-1)),dim=-1)
        results = self.renderer.forward(verts)
        renders = (results.detach().cpu().numpy().transpose((0,2,3,1))*256).astype(np.uint8)[:,:,:,::-1]

        render_mask = ~(renders>100)#.astype(np.bool) 去除渲染结果（白底时）的黑色毛刺边
        renders[render_mask] = images[render_mask]

        return renders

    def visulize_result(self,outputs,kps,data,name,vnum = 6, white_background=False,rtype='',nokp=False,org_name=True,org_img=False,keep_name=False):
        if not keep_name:
            if 'name' in data:
                img_names = data['name']
            else:
                img_names = data['imgpath']
        imgs = data['image_org'].contiguous().numpy().astype(np.uint8)[:vnum,:,:,::-1]
        vnum = imgs.shape[0]
        if self.high_resolution:
            kps = ((kps.detach().contiguous().cpu().numpy()+1)/2 * 500).reshape(-1,14,2)[:vnum]
        else:
            kps = ((kps.detach().contiguous().cpu().numpy()+1)/2 * imgs.shape[1]).reshape(-1,14,2)[:vnum]

        kp_imgs = []
        #white_background=False
        for idx in range(vnum):
            if white_background:
                kp_imgs.append(draw_lsp_14kp__bone(np.ones_like(imgs[idx])*255, kps[idx]))
            else:
                kp_imgs.append(draw_lsp_14kp__bone(imgs[idx].copy(), kps[idx]))

        ((cam,pose,shape), predict_verts, predict_j2d, predict_j3d, predict_Rs,verts_camed,j3d_camed) = outputs
        if white_background:
            rendered_imgs = self.visualize_renderer(verts_camed[:vnum], np.ones_like(imgs)*255)
        else:
            rendered_imgs = self.visualize_renderer(verts_camed[:vnum], imgs)

        if org_img:
            offsets = data['offsets'].numpy()
            org_image_names = data['imgpath']
            #image_org = data['org_image'].numpy()
            imgs = []
            #imgs = data['orgimage'].numpy()
            org_image = []
            for n in range(rendered_imgs.shape[0]):
                org_imge = cv2.imread(org_image_names[n])#image_org[n].numpy().astype(np.uint8)
                imgs.append(org_imge.copy())
                resized_images = cv2.resize(rendered_imgs[n], (offsets[n,0]+1, offsets[n,1]+1), interpolation = cv2.INTER_CUBIC)
                #print(offsets[n,2],(offsets[n,3]-1),offsets[n,4],(offsets[n,5]-1))
                org_imge[offsets[n,2]:(offsets[n,3]-1),offsets[n,4]:(offsets[n,5]-1),:] = resized_images[offsets[n,6]:(offsets[n,7]-1+offsets[n,6]),offsets[n,8]:(offsets[n,9]+offsets[n,8]-1),:]
                org_image.append(org_imge)

            #imgs = np.array(imgs)
            #org_image = np.array(org_image)

        for idx in range(vnum):
            if nokp:
                if org_img:
                    if len(org_image[idx].shape)<3:
                        print(org_image_names[idx],org_image[idx].shape)
                        continue
                    result_img = np.hstack((imgs[idx], org_image[idx]))
                else:
                    result_img = np.hstack((imgs[idx], rendered_imgs[idx]))
            else:
                result_img = np.hstack((imgs[idx],kp_imgs[idx], rendered_imgs[idx]))
            #cv2.imwrite(name+'_{}_org_{}.jpg'.format(idx,rtype),imgs[idx])
            if keep_name:
                #print(name[idx])
                cv2.imwrite(name[idx],result_img)
            elif org_name:
                cv2.imwrite('{}{}-{}'.format(name.split(os.path.basename(name))[0],img_names[idx].split('/')[-2],os.path.basename(img_names[idx])),result_img)
            else:
                cv2.imwrite(name+'_{}_{}.jpg'.format(idx,rtype),result_img)


    def render_video(self,verts,params,images,org_image,offsets,name):
        rendered_images = self.visualize_renderer(verts,params[:,:3],images)

        for n in range(verts.shape[0]):
            resized_images = cv2.resize(rendered_images[n], (offsets[n,0]+1, offsets[n,1]+1), interpolation = cv2.INTER_CUBIC)
            org_image[n,offsets[n,2]:(offsets[n,3]-1),offsets[n,4]:(offsets[n,5]-1),:] = resized_images[offsets[n,6]:(offsets[n,7]-1+offsets[n,6]),offsets[n,8]:(offsets[n,9]+offsets[n,8]-1),:]
        self.make_mp4(org_image,name)

    def make_mp4(self,images,name):
        num = images.shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_movie = cv2.VideoWriter(name+'.mp4', fourcc, 50, (images.shape[2], images.shape[1]))
        for i in range(num):
            if i%100==0:
                print('Writing frame: ',i,'/',num)
            output_movie.write(images[i])