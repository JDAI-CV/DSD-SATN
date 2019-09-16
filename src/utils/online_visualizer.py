from queue import Queue
import numpy as np
import sys
import time
import os
import torch
import torch.multiprocessing as mp
from multiprocessing import Process
from multiprocessing import Queue as pQueue
from threading import Thread
import json
import cv2
sys.path.append('../')
import config
#from fast_rendering.face3d import mesh

class Visualizer_online:
    def __init__(self, save_video=True, vis=True,
                savepath='./result.avi', fourcc=cv2.VideoWriter_fourcc(*'XVID'), fps=30, frameSize=(640,480),
                queueSize=1024, show_size=512):
        if save_video:
            # initialize the file video stream along with the boolean
            # used to indicate if the thread should be stopped or not
            self.stream = cv2.VideoWriter(savepath, fourcc, fps, frameSize)
            assert self.stream.isOpened(), 'Cannot open video for writing'
        self.save_video = save_video
        self.vis = vis
        self.fps = fps
        self.stopped = False
        self.final_result = []
        self.show_size = show_size
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)
        self.cache = Queue(maxsize=queueSize)

        with open(os.path.join(config.project_dir,'model/neutral_smpl_with_cocoplus_reg.txt'), 'r') as reader:
            faces = json.load(reader)['f']
        self.triangles = np.array(faces)

        self.s = 180 * show_size/256#/(np.max(vertices[:,1]) - np.min(vertices[:,1]))*2
        self.R = mesh.transform.angle2matrix([0, 0, 0])
        self.t = [0, 0.7, 0]
        self.h = self.w = show_size
        print('Initialized Visualizer Online!')
        

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.organizer, args=())
        t.daemon = True
        t.start()

        t2 = Thread(target=self.update, args=())
        t2.daemon = True
        t2.start()
        return self

    def rendering(self,vertices):
        vertices[:,2] *= -1
        vertices[:,0] *= -1

        self.s = 180 * self.show_size/256 /(np.max(vertices[:,1]) - np.min(vertices[:,1]))
        #print(np.max(vertices[:,1]) , np.min(vertices[:,1]),np.max(vertices[:,1]) - np.min(vertices[:,1]))
        #self.s = 180/(np.max(vertices[:,1]) - np.min(vertices[:,1]))
        vertices = mesh.transform.similarity_transform(vertices, self.s, self.R, self.t)
        image_vertices = mesh.transform.to_image(vertices, self.h, self.w)

        z = image_vertices[:,2:]
        z = z - np.min(z)
        z = z/np.max(z)
        attribute = z

        depth_image = mesh.render.render_colors(image_vertices, self.triangles, attribute, self.h, self.w, c=1)
        depth_image = np.squeeze(depth_image)
        depth_image[depth_image>1] = 1
        #print(depth_image[depth_image>1], depth_image[depth_image<-1])
        #io.imsave('./depth.jpg', np.squeeze(depth_image))
        return depth_image

    def organizer(self):
        while True:
            if self.stopped:
                return 
            if not self.cache.empty():
                (predict_verts,org_imgs,orig_imgs,frame_nums,boxes,name_base) = self.cache.get()
                for i in range(len(predict_verts)):
                    predict_vert,org_img,orig_img,box = predict_verts[i].detach().cpu().numpy(), org_imgs[i], orig_imgs[i],boxes[i]
                    cv2.rectangle(orig_img, tuple(box[:2]), tuple(box[2:]), (255,0,0), 2)
                    h,w,_ = orig_img.shape
                    h2,w2,_  = org_img.shape
                    show_img = np.zeros((h,w+self.w,3),dtype = np.uint8)
                    show_img[:,:w] = orig_img
                    show_img[:h2,w:w+w2] = org_img
                    result_img = self.rendering(predict_vert)*255
                    show_img[h2:h2+self.h,w:w+self.w] = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)

                    name = name_base+'-{}.jpg'.format(i)
                    self.Q.put((show_img,name))

            else:
                time.sleep(0.01)

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                if self.save_video:
                    self.stream.release()
                return
            # otherwise, ensure the queue is not empty
            if not self.Q.empty():
                img,name = self.Q.get()

                if self.save_video or self.vis:
                    if self.vis:
                        cv2.imwrite(name,img)
                        #cv2.imshow("Online Demo", img)
                        cv2.waitKey(int(1000/(self.len()+1)))

                    if self.save_video:
                        self.stream.write(img)

            else:
                time.sleep(0.01)

    def running(self):
        # indicate that the thread is still running
        time.sleep(0.2)
        return not self.Q.empty()

    def add(self, predict_verts,org_imgs,orig_imgs,frame_nums,boxes,name='test.jpg'):
        self.cache.put((predict_verts,org_imgs,orig_imgs,frame_nums,boxes,name))
        # save next frame in the queue
        #self.Q.put((boxes, scores, hm_data, pt1, pt2, orig_img, im_name))

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        time.sleep(0.2)

    def results(self):
        # return final result
        return self.final_result

    def len(self):
        # return queue len
        return self.Q.qsize()

def main():
    repeat_num=2
    v = Visualizer_online(vis=True, frameSize=(512+256,512))
    v.start()
    vertices = torch.from_numpy(np.load('vertices.npy')).unsqueeze(0).repeat(repeat_num,1,1).cuda()

    v.add(vertices, np.ones((repeat_num,256,256,3),dtype=np.uint8)*155, np.ones((repeat_num,512,512,3),dtype=np.uint8)*255, [0,1], [[22,33,222,222],[22,33,500,500]])
    time.sleep(1)
    v.stop()

if __name__ == '__main__':
    main()
