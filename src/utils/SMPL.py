#encoding=utf-8
import torch
import sys
import os
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('utils/',''))
from config import args
import json
import sys
import numpy as np
from utils.util import *
import torch.nn as nn
import h5py
import torch.nn.functional as F


class SMPL(nn.Module):
    def __init__(self, model_path, joint_type = 'cocoplus', obj_saveable = False,batch_size = 200):
        super(SMPL, self).__init__()

        if joint_type not in ['cocoplus', 'lsp']:
            msg = 'unknow joint type: {}, it must be either "cocoplus" or "lsp"'.format(joint_type)
            sys.exit(msg)

        self.model_path = model_path
        self.joint_type = joint_type
        with open(self.model_path, 'r') as reader:
            model = json.load(reader)
        self.faces = model['f']
        np_v_template = np.array(model['v_template'], dtype = np.float)

        self.register_buffer('v_template', torch.from_numpy(np_v_template).float())
        self.size = [np_v_template.shape[0], 3]
        np_shapedirs = np.array(model['shapedirs'], dtype = np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        self.register_buffer('shapedirs', torch.from_numpy(np_shapedirs).float())
        np_J_regressor = np.array(model['J_regressor'], dtype = np.float)
        self.register_buffer('J_regressor', torch.from_numpy(np_J_regressor).float())

        np_posedirs = np.array(model['posedirs'], dtype = np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', torch.from_numpy(np_posedirs).float())
        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
        with open(args.coco25_regressor_path, 'rb') as f:
            dd = pickle.load(f, encoding='latin1')
            coco25_2_lsp14 = [24, 1,2, 3,4,23, 6,7,8,9,10,11,12,13]
            regressor = dd['cocoplus_regressor'][coco25_2_lsp14]
            np_joint_regressor = np.array(regressor.todense(), dtype = np.float).T
        #np_joint_regressor = np.array(model['cocoplus_regressor'], dtype = np.float)

        if joint_type == 'lsp':
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor[:, :14]).float())
        else:
            self.register_buffer('joint_regressor', torch.from_numpy(np_joint_regressor).float())
        np_weights = np.array(model['weights'], dtype = np.float)

        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]

        np_weights = np.tile(np_weights, (batch_size, 1))
        self.register_buffer('weight', torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component))

        self.register_buffer('e3', torch.eye(3).float())

        self.cur_device = None

    def save_obj(self, verts, obj_mesh_name):
        if not self.faces:
            msg = 'obj not saveable!'
            sys.exit(msg)

        with open(obj_mesh_name, 'w') as fp:
            for v in verts:
                fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

            for f in self.faces: # Faces are 1-based, not 0-based in obj files
                fp.write( 'f %d %d %d\n' %  (f[0] + 1, f[1] + 1, f[2] + 1) )

    def forward(self, beta, param, get_skin = False,rotate_base=False,root_rot_mat = None,get_org_joints = False):
        if not self.cur_device:
            device = beta.device
            self.cur_device = torch.device(device.type, device.index)

        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        #[num_batch*6890] * [6890*24] = num_batch*24
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim = 2)
        Rs = batch_rodrigues(param.view(-1, 3)).view(-1, 24, 3, 3)
        pose_feature = (Rs[:, 1:, :, :]).sub(1.0, self.e3).view(-1, 207)
        v_posed = torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1]) + v_shaped
        self.J_transformed, A = batch_global_rigid_transformation(Rs, J, self.parents, rotate_base = rotate_base,root_rot_mat =root_rot_mat)

        while num_batch>self.weight.shape[0]:
            print('stacking SMPLweight to facilitate larger batch_size!')
            self.weight = torch.cat((self.weight,self.weight),0)
        weight = self.weight[:num_batch]
        W = weight.view(num_batch, -1, 24)
        T = torch.matmul(W, A.view(num_batch, 24, 16)).view(num_batch, -1, 4, 4)

        v_posed_homo = torch.cat([v_posed, torch.ones(num_batch, v_posed.shape[1], 1, device = self.cur_device)], dim = 2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]

        joint_x = torch.matmul(verts[:, :, 0], self.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.joint_regressor)

        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        if get_org_joints:
            return verts, self.J_transformed, Rs

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def reverse_derive_rotation(self, beta, param, rotated_kp3d_org = None):
        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, torch.ones(num_batch, 1, 1).cuda()], dim = 1)
            return torch.cat([R_homo, t_homo], 2)

        num_batch = beta.shape[0]
        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        #[num_batch*6890] * [6890*24] = num_batch*24
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)

        J = torch.stack([Jx, Jy, Jz], dim = 2)
        J = torch.unsqueeze(J, -1)
        Js_w0 = torch.cat([J, torch.zeros(num_batch, 24, 1, 1)], dim = 2)
        J = F.pad(Js_w0, [3, 0, 0, 0, 0, 0, 0, 0])

        rotated_kp3d_org = torch.cat([rotated_kp3d_org.unsqueeze(-1), torch.zeros(num_batch, 24, 1, 1)], dim = 2)
        rotated_kp3d_org = F.pad(rotated_kp3d_org, [3, 0, 0, 0, 0, 0, 0, 0])

        Rs = batch_rodrigues(param.view(-1, 3)).view(-1, 24, 3, 3)
        root_rotation = Rs[:, 0, :, :]
        A0 = make_A(root_rotation,J[:,0])
        retrieve_rotations = []
        for i in range(1,4):
            assert self.parents[i] == 0
            j_here = J[:, i] - J[:, self.parents[i]]
            j_here[:,-1,-1]=1.
            j_invs = []
            for eachj in j_here:
                j_inv = torch.inverse(eachj)
                j_invs.append(j_inv)
            j_invs = torch.stack(j_invs,dim=0)
            print(rotated_kp3d_org[:,i].unsqueeze(-1).shape, J[:,0].shape, j_invs.shape)
            retrieve_rotation = []
            for num_idx in range(rotated_kp3d_org.shape[0]):
                retrieve_rotation.append(torch.matmul(rotated_kp3d_org[num_idx,i],j_invs[num_idx]))
            retrieve_rotations.append(retrieve_rotation)

        print(retrieve_rotations)
        print(torch.abs(retrieve_rotations[0][0]-retrieve_rotations[1][0]).sum())
        return retrieve_rotations

if __name__ == '__main__':
    device = torch.device('cuda', 0)
    smpl = SMPL(args.smpl_model, joint_type = 'lsp', obj_saveable = True).to(device)
    #import utils.neuralrenderer_render as neuralrenderer
    #renderer = neuralrenderer.get_renderer().cuda()
    os.makedirs('test',exist_ok=True)
    for test_joint_idx in [3]:
        pose= np.array([
                    0,   0,   0,
                    -1.20581151e-03,   8.60930011e-02,   4.45963144e-02,
                    -1.52801601e-02,  -1.16911056e-02,  -6.02894090e-03,
                    1.62427306e-01,   4.26302850e-02,  -1.55304456e-02,
                    2.58729942e-02,  -2.15941742e-01,  -6.59851432e-02,
                    7.79098943e-02,   1.96353287e-01,   6.44420758e-02,
                    -5.43042570e-02,  -3.45508829e-02,   1.13200583e-02,
                    -5.60734887e-04,   3.21716577e-01,  -2.18840033e-01,
                    -7.61821344e-02,  -3.64610642e-01,   2.97633410e-01,
                    9.65453908e-02,  -5.54007106e-03,   2.83410680e-02,
                    -9.57194716e-02,   9.02515948e-02,   3.31488043e-01,
                    -1.18847653e-01,   2.96623230e-01,  -4.76809204e-01,
                    -1.53382001e-02,   1.72342166e-01,  -1.44332021e-01,
                    -8.10869411e-02,   4.68325168e-02,   1.42248288e-01,
                    -4.60898802e-02,  -4.05981280e-02,   5.28727695e-02,
                    3.20133418e-02,  -5.23784310e-02,   2.41559884e-03,
                    -3.08033824e-01,   2.31431410e-01,   1.62540793e-01,
                    6.28208935e-01,  -1.94355965e-01,   7.23800480e-01,
                    -6.49612308e-01,  -4.07179184e-02,  -1.46422181e-02,
                    4.51475441e-01,   1.59122205e+00,   2.70355493e-01,
                    2.04248756e-01,  -6.33800551e-02,  -5.50178960e-02,
                    -1.00920045e+00,   2.39532292e-01,   3.62904727e-01,
                    -3.38783532e-01,   9.40650925e-02,  -8.44506770e-02,
                    3.55101633e-03,  -2.68924050e-02,   4.93676625e-02],dtype = np.float)

        beta = np.array([-0.25349993,  0.25009069,  0.21440795,  0.78280628,  0.08625954,
                    0.28128183,  0.06626327, -0.26495767,  0.09009246,  0.06537955 ])
        sample_list = list(np.pi*np.array(list(range(-2,2,1)))/8)
        for i in sample_list:
            for j in sample_list:
                for k in sample_list:
                    pose[test_joint_idx*3:(test_joint_idx+1)*3] = [i,j,k]
                    vbeta = torch.tensor(np.array([beta])).float().to(device)
                    vpose = torch.tensor(np.array([pose])).float().to(device)
                    verts, joint, r = smpl(vbeta, vpose, get_skin = True)
                    smpl.save_obj(verts[0].cpu().numpy(), 'test/{}_{}_{}_{}.obj'.format(test_joint_idx,i,j,k))

