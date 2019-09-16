import torch.nn as nn
import sys
from torch.utils.data import Dataset, DataLoader
import scipy.io as scio
import os
import glob
import numpy as np
import random
import cv2
import json
import h5py
import torch
import math
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('utils/',''))
from utils.util import *
from utils.SMPL import SMPL
import neural_renderer as nr

class Renderer_seg(nn.Module):
    def __init__(self, faces,np_v_template = None,test=False,test_camera=False,high_resolution=False):
        super(Renderer_seg, self).__init__()
        image_size=500 if high_resolution else 256

        self.camera_distance = 1/math.tan(math.pi/18)#3.464#1.732
        self.elevation = 0
        self.azimuth = 0
        self.register_buffer('faces', faces)
        self.renderer = nr.Renderer(image_size = image_size,camera_mode='look_at',\
            light_intensity_ambient=0.72, light_intensity_directional=0.3,\
            light_color_ambient=[1,1,1], light_color_directional=[1,1,1],\
            light_direction=[0,1,0]).cuda()

        texture_size = 4
        textures = torch.ones(64, self.faces.shape[1], texture_size, texture_size, texture_size, 3, dtype=torch.float32)
        textures[:,:,:,:,:] = torch.from_numpy(np.array([0.7098039, 0.84117647, 0.95882353])).float()#'light_blue'193,210,240;0.65098039, 0.74117647, 0.85882353;;'light_pink': [.9, .7, .7]
        self.register_buffer('textures', textures.cuda())
        self.renderer.viewing_angle = 10
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance,self.elevation,self.azimuth)
        #self.renderer.camera_direction = [0,0,1]

        #testing
        if test:
            v_template = torch.from_numpy(np_v_template).float().cuda()
            #v_template = np_v_template.float().cuda()
            projected_seg = self.forward(v_template)
            projected_seg = (projected_seg.cpu().numpy()[0]*255).astype(np.uint8)
            cv2.imwrite('./cut_{}_{}_{}.png'.format(self.azimuth,self.elevation,self.camera_distance),projected_seg)
        if test_camera:
            v_template = torch.from_numpy(np_v_template).float().cuda()
            for self.camera_distance in range(1,3):
                for self.elevation in range(0,90,30):
                    for self.azimuth in range(0,360,30):
                        self.renderer.eye = nr.get_points_from_angles(self.camera_distance,self.elevation,self.azimuth)
                        projected_seg = self.forward(v_template)
                        projected_seg = projected_seg.cpu().numpy()[0].astype(np.bool).astype(np.int)*255
                        cv2.imwrite('./cut_{}_{}_{}.png'.format(self.azimuth,self.elevation,self.camera_distance),projected_seg)

    def forward(self, vertices):
        N = vertices.shape[0]
        vertices[:,:,1] *=-1
        image = self.renderer(vertices, self.faces[:N], self.textures[:N],mode='rgb')
        return image

    def standard(self,vertices):
        one_array = torch.ones_like(vertices)
        zero_array = torch.zeros_like(vertices)
        vertices = torch.where(vertices>1,one_array,vertices)
        vertices = torch.where(vertices<0,zero_array,vertices)
        return vertices

def make_reference_image(filename_ref, filename_obj,org_image):
    model = Renderer_seg()
    model.cuda()
    images = model.renderer.render(model.vertices, model.faces, torch.tanh(model.textures))
    image = images.detach().cpu().numpy()[0]
    image = image.transpose((1,2,0))
    print(image.shape)

    org = cv2.imread(org_image)
    org = cv2.resize(org,(256,256))
    shape = org.shape
    for i in range(shape[0]):
        for j in range(shape[1]):

            if (image[i,j]==0).sum()==3:
                image[i,j,0]=org[i,j,2]
                image[i,j,1]=org[i,j,1]
                image[i,j,2]=org[i,j,0]
            else:
                image[i,j] = (image[i,j])*256

    imsave(filename_ref, image.astype(np.uint8))

def get_renderer(test=False,test_camera=False,face_num=128,high_resolution=False):
    with open(args.smpl_model, 'r') as reader:
        model = json.load(reader)

    faces = model['f']
    faces = torch.from_numpy(np.array([faces])).repeat(face_num,1,1).int().cuda()
    np_v_template = np.array([model['v_template']])
    renderer = Renderer_seg(faces,np_v_template,test=test,test_camera=test_camera,high_resolution=high_resolution)

    return renderer

if __name__ == '__main__':
    get_renderer(test=True)
