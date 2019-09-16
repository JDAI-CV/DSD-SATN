#encoding=utf-8
import h5py
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(__file__).replace(os.path.basename(__file__),'').replace('utils/',''))
from config import args
import json
import torch.nn.functional as F
import cv2
import math
from scipy import interpolate
import hashlib
import shutil
import pickle
import csv

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import io

# logger tools

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

def wrap(func, *args, unsqueeze=False):
    """
    Wrap a torch function so it can be called with NumPy arrays.
    Input and return types are seamlessly converted.
    """
    # Convert input types where applicable
    args = list(args)
    for i, arg in enumerate(args):
        if type(arg) == np.ndarray:
            args[i] = torch.from_numpy(arg)
            if unsqueeze:
                args[i] = args[i].unsqueeze(0)

    result = func(*args)

    # Convert output types where applicable
    if isinstance(result, tuple):
        result = list(result)
        for i, res in enumerate(result):
            if type(res) == torch.Tensor:
                if unsqueeze:
                    res = res.squeeze(0)
                result[i] = res.numpy()
        return tuple(result)
    elif type(result) == torch.Tensor:
        if unsqueeze:
            result = result.squeeze(0)
        return result.numpy()
    else:
        return result

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

# Math transform

def compute_similarity_transform(S1, S2):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_average_loss(loss_list):
    loss_np = np.array(loss_list)
    loss = np.mean(loss_np,axis=0)
    return loss

def _init_weights_deconv(m):
    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    m.weight.data.normal_(0, math.sqrt(2. / n))

def _init_batchnorm(m):
    m.weight.data.fill_(1)
    m.bias.data.zero_()

def load_mean_param():
    mean = np.zeros(args.total_param_count, dtype = np.float)

    mean_values = h5py.File(args.smpl_mean_param_path)
    mean_pose = mean_values['pose']
    mean_pose[:3] = 0
    mean_shape = mean_values['shape']
    mean_pose[0]=np.pi

    #init scale is 0.9
    mean[0] = 0.9

    mean[3:75] = mean_pose[:]
    mean[75:] = mean_shape[:]

    return mean

def batch_rodrigues(param):
    #param N x 3
    batch_size = param.shape[0]

    l1norm = torch.norm(param + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(param, angle)
    angle = angle * 0.5

    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)

    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)

    return quat2mat(quat)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat

def batch_global_rigid_transformation(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x = torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float().cuda()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1).cuda()], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1).cuda()], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_global_rigid_transformation_cpu(Rs, Js, parent, rotate_base = False,root_rot_mat =None):
    N = Rs.shape[0]
    if rotate_base:
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype = np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    elif root_rot_mat is not None:
        np_rot_x = np.reshape(np.tile(root_rot_mat, [N, 1]), [N, 3, 3])
        rot_x =torch.from_numpy(np_rot_x).float()
        root_rotation = torch.matmul(Rs[:, 0, :, :],  rot_x)
    else:
        root_rotation = Rs[:, 0, :, :]
    Js = torch.unsqueeze(Js, -1)

    def make_A(R, t):
        R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
        t_homo = torch.cat([t, torch.ones(N, 1, 1)], dim = 1)
        return torch.cat([R_homo, t_homo], 2)

    A0 = make_A(root_rotation, Js[:, 0])
    results = [A0]

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)
        results.append(res_here)

    results = torch.stack(results, dim = 1)

    new_J = results[:, :, :3, 3]
    Js_w0 = torch.cat([Js, torch.zeros(N, 24, 1, 1)], dim = 2)
    init_bone = torch.matmul(results, Js_w0)
    init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
    A = results - init_bone

    return new_J, A

def batch_lrotmin(param):
    param = param[:,3:].contiguous()
    Rs = batch_rodrigues(param.view(-1, 3))
    print(Rs.shape)
    e = torch.eye(3).float()
    Rs = Rs.sub(1.0, e)

    return Rs.view(-1, 23 * 9)

def batch_orth_proj(X, camera, mode='2d'):
    camera = camera.view(-1, 1, 3)
    s = camera[:, :, 0].unsqueeze(-1)
    X_trans = X[:,:,:2].contiguous()
    if mode=='2d':
        X_trans = s * X_trans + camera[:, :, 1:]
        return X_trans
    elif mode=='v3d':
        X[:, :, :2] = s * X_trans + camera[:, :, 1:]
        return X
    elif mode=='j3d':
        X[:, :, :2] = s * X_trans/torch.abs(s) + camera[:, :, 1:]
        return X
    else:
        print('projection mode is not included')
        return X

def calc_aabb(ptSets):

    ptLeftTop     = np.array([np.min(ptSets[:,0]),np.min(ptSets[:,1])])
    ptRightBottom = np.array([np.max(ptSets[:,0]),np.max(ptSets[:,1])])
    return [ptLeftTop, ptRightBottom]

    ptLeftTop     = np.array([ptSets[0][0], ptSets[0][1]])
    ptRightBottom = ptLeftTop.copy()
    for pt in ptSets:
        ptLeftTop[0]     = min(ptLeftTop[0], pt[0])
        ptLeftTop[1]     = min(ptLeftTop[1], pt[1])
        ptRightBottom[0] = max(ptRightBottom[0], pt[0])
        ptRightBottom[1] = max(ptRightBottom[1], pt[1])

    return ptLeftTop, ptRightBottom#, len(ptSets) >= 5

def calc_aabb_batch(ptSets_batch):
    batch_size = ptSets_batch.shape[0]
    ptLeftTop     = np.array([np.min(ptSets_batch[:,:,0],axis=1),np.min(ptSets_batch[:,:,1],axis=1)]).T
    ptRightBottom = np.array([np.max(ptSets_batch[:,:,0],axis=1),np.max(ptSets_batch[:,:,1],axis=1)]).T
    bbox = np.concatenate((ptLeftTop.reshape(batch_size,1,2),ptRightBottom.reshape(batch_size,1,2)),axis=1)
    return bbox

def calc_obb(ptSets):
    ca = np.cov(ptSets,y = None,rowvar = 0,bias = 1)
    v, vect = np.linalg.eig(ca)
    tvect = np.transpose(vect)
    ar = np.dot(ptSets,np.linalg.inv(tvect))
    mina = np.min(ar,axis=0)
    maxa = np.max(ar,axis=0)
    diff    = (maxa - mina)*0.5
    center  = mina + diff
    corners = np.array([center+[-diff[0],-diff[1]],center+[diff[0],-diff[1]],center+[diff[0],diff[1]],center+[-diff[0],diff[1]]])
    corners = np.dot(corners, tvect)
    return corners[0], corners[1], corners[2], corners[3]

def get_image_cut_box(leftTop, rightBottom, ExpandsRatio, Center = None):
    try:
        l = len(ExpandsRatio)
    except:
        ExpandsRatio = [ExpandsRatio, ExpandsRatio, ExpandsRatio, ExpandsRatio]

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]

        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    if Center == None:
        Center = (leftTop + rightBottom) // 2

    Center, leftTop, rightTop, rightBottom, leftBottom = _expand_crop_box(leftTop, rightBottom, ExpandsRatio)
    offset = (rightBottom - leftTop) // 2

    cx = offset[0]
    cy = offset[1]

    r = max(cx, cy)

    cx = r
    cy = r

    x = int(Center[0])
    y = int(Center[1])

    return [x - cx, y - cy], [x + cx, y + cy]

def shrink(leftTop, rightBottom, width, height):
    xl = -leftTop[0]
    xr = rightBottom[0] - width

    yt = -leftTop[1]
    yb = rightBottom[1] - height

    cx = (leftTop[0] + rightBottom[0]) / 2
    cy = (leftTop[1] + rightBottom[1]) / 2

    r = (rightBottom[0] - leftTop[0]) / 2

    sx = max(xl, 0) + max(xr, 0)
    sy = max(yt, 0) + max(yb, 0)

    if (xl <= 0 and xr <= 0) or (yt <= 0 and yb <=0):
        return leftTop, rightBottom
    elif leftTop[0] >= 0 and leftTop[1] >= 0 : # left top corner is in box
        l = min(yb, xr)
        r = r - l / 2
        cx = cx - l / 2
        cy = cy - l / 2
    elif rightBottom[0] <= width and rightBottom[1] <= height : # right bottom corner is in box
        l = min(yt, xl)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy + l / 2
    elif leftTop[0] >= 0 and rightBottom[1] <= height : #left bottom corner is in box
        l = min(xr, yt)
        r = r - l  / 2
        cx = cx - l / 2
        cy = cy + l / 2
    elif rightBottom[0] <= width and leftTop[1] >= 0 : #right top corner is in box
        l = min(xl, yb)
        r = r - l / 2
        cx = cx + l / 2
        cy = cy - l / 2
    elif xl < 0 or xr < 0 or yb < 0 or yt < 0:
        return leftTop, rightBottom
    elif sx >= sy:
        sx = max(xl, 0) + max(0, xr)
        sy = max(yt, 0) + max(0, yb)
        # cy = height / 2
        if yt >= 0 and yb >= 0:
            cy = height / 2
        elif yt >= 0:
            cy = cy + sy / 2
        else:
            cy = cy - sy / 2
        r = r - sy / 2

        if xl >= sy / 2 and xr >= sy / 2:
            pass
        elif xl < sy / 2:
            cx = cx - (sy / 2 - xl)
        else:
            cx = cx + (sy / 2 - xr)
    elif sx < sy:
        cx = width / 2
        r = r - sx / 2
        if yt >= sx / 2 and yb >= sx / 2:
            pass
        elif yt < sx / 2:
            cy = cy - (sx / 2 - yt)
        else:
            cy = cy + (sx / 2 - yb)


    return [cx - r, cy - r], [cx + r, cy + r]

def off_set_pts(keyPoints, leftTop):
    result = keyPoints.copy()
    result[:, 0] -= leftTop[0]
    result[:, 1] -= leftTop[1]
    return result

'''
    cut the image, by expanding a bounding box
'''
def cut_image(originImage, kps, expand_ratio, leftTop, rightBottom,cam=None,centralize=False):

    original_shape = originImage.shape
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    leftTop[0] = max(0, leftTop[0])
    leftTop[1] = max(0, leftTop[1])

    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    leftTop      = np.array([int(leftTop[0]), int(leftTop[1])])
    rightBottom  = np.array([int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)])

    length = max(rightBottom[1] - leftTop[1]+1, rightBottom[0] - leftTop[0]+1)
    if length<20:
        return False,False,False

    dstImage = np.zeros(shape = [length,length, channels], dtype = np.uint8)
    dstImage[:,:,:] = 0

    offset = np.array([lt[0] - leftTop[0], lt[1] - leftTop[1]])
    size   = [rb[0] - lt[0], rb[1] - lt[1]]

    try:
        dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    except Exception as error:
        return False,False,False

    if cam is not None:
        cam[1] = (cam[1]+1.0)*float(original_shape[1])/float(length)-2.0*float(leftTop[0])/float(length)-1.0
        cam[2] = (cam[2]+1.0)*float(original_shape[0])/float(length)-2.0*float(leftTop[1])/float(length)-1.0
        cam[0] *= original_shape[0]/length

        return dstImage, off_set_pts(kps, leftTop),cam,(offset,lt,rb,size,original_shape[:2])

    return dstImage, off_set_pts(kps, leftTop),(offset,lt,rb,size,original_shape[:2])

def getltrb(expand_ratio, leftTop, rightBottom,height,width,kp2d):
    inimage = (kp2d<0).sum()
    inimage += (kp2d[:,0]>320).sum()
    inimage += (kp2d[:,1]>240).sum()
    if inimage>0:
        return True
    originImage = np.zeros((240,320,3))
    original_shape = originImage.shape
    height       = originImage.shape[0]
    width        = originImage.shape[1]
    channels     = originImage.shape[2] if len(originImage.shape) >= 3 else 1
    leftTop, rightBottom = get_image_cut_box(leftTop, rightBottom, expand_ratio)

    lt = [int(leftTop[0]), int(leftTop[1])]
    rb = [int(rightBottom[0]), int(rightBottom[1])]

    lt[0] = max(0, lt[0])
    lt[1] = max(0, lt[1])
    rb[0] = min(rb[0], width)
    rb[1] = min(rb[1], height)

    h = float(rb[1]-lt[1])
    w = float(rb[0]-lt[0])

    leftTop      = [int(leftTop[0]), int(leftTop[1])]
    rightBottom  = [int(rightBottom[0] + 0.5), int(rightBottom[1] + 0.5)]

    length = max(rightBottom[1] - leftTop[1]+1, rightBottom[0] - leftTop[0]+1)

    dstImage = np.zeros(shape = [length,length, channels], dtype = np.uint8)
    dstImage[:,:,:] = 0

    offset = [lt[0] - leftTop[0], lt[1] - leftTop[1]]
    size   = [rb[0] - lt[0], rb[1] - lt[1]]

    try:
        dstImage[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0], :] = originImage[lt[1]:rb[1], lt[0]:rb[0],:]
    except:
        print('error in image crop')
        return True
    mask = np.ones((240,320))

    if mask is not None:
        dstmask = np.zeros(shape = [length, length], dtype = np.uint8)
        dstmask[:,:] = 0
        try:
            dstmask[offset[1]:size[1] + offset[1], offset[0]:size[0] + offset[0]] = mask[lt[1]:rb[1], lt[0]:rb[0]]
        except:
            print('error in mask crop')
            return True

    if h<4 or w<4:
        return True
    else:
        return False

def reflect_lsp_kp(kps):
    kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
    joint_ref = kps[kp_map]
    joint_ref[:,0] = -joint_ref[:,0]

    return joint_ref - np.mean(joint_ref, axis = 0)

def reflect_pose(poses):
    swap_inds = np.array([
            0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, 15, 16, 17, 12, 13, 14, 18,
            19, 20, 24, 25, 26, 21, 22, 23, 27, 28, 29, 33, 34, 35, 30, 31, 32,
            36, 37, 38, 42, 43, 44, 39, 40, 41, 45, 46, 47, 51, 52, 53, 48, 49,
            50, 57, 58, 59, 54, 55, 56, 63, 64, 65, 60, 61, 62, 69, 70, 71, 66,
            67, 68
    ])

    sign_flip = np.array([
            1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
            -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1,
            -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1,
            1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1, -1, 1, -1,
            -1, 1, -1, -1
    ])

    return poses[swap_inds] * sign_flip

def crop_image(image_path, angle, lt, rb, scale, kp_2d, crop_size):
    '''
        given a crop box, expand it at 4 directions.(left, right, top, bottom)
    '''
    assert 'error algorithm exist.' and 0

    def _expand_crop_box(lt, rb, scale):
        center = (lt + rb) / 2.0
        xl, xr, yt, yb = lt[0] - center[0], rb[0] - center[0], lt[1] - center[1], rb[1] - center[1]
        xl, xr, yt, yb = xl * scale[0], xr * scale[1], yt * scale[2], yb * scale[3]
        #expand it
        lt, rb = np.array([center[0] + xl, center[1] + yt]), np.array([center[0] + xr, center[1] + yb])
        lb, rt = np.array([center[0] + xl, center[1] + yb]), np.array([center[0] + xr, center[1] + yt])
        center = (lt + rb) / 2
        return center, lt, rt, rb, lb

    def _extend_box(center, lt, rt, rb, lb, crop_size):
        lx, ly = np.linalg.norm(rt - lt), np.linalg.norm(lb - lt)
        dx, dy = (rt - lt) / lx, (lb - lt) / ly
        l = max(lx, ly) / 2.0
        return center - l * dx - l * dy, center + l * dx - l *dy, center + l * dx + l * dy, center - l * dx + l * dy, dx, dy, crop_size * 1.0 / l

    def _get_sample_points(lt, rt, rb, lb, crop_size):
        vec_x = rt - lt
        vec_y = lb - lt
        i_x, i_y = np.meshgrid(range(crop_size), range(crop_size))
        i_x = i_x.astype(np.float)
        i_y = i_y.astype(np.float)
        i_x /= float(crop_size)
        i_y /= float(crop_size)
        interp_points = i_x[..., np.newaxis].repeat(2, axis=2) * vec_x + i_y[..., np.newaxis].repeat(2, axis=2) * vec_y
        interp_points += lt
        return interp_points

    def _sample_image(src_image, interp_points):
        sample_method = 'nearest'
        interp_image = np.zeros((interp_points.shape[0] * interp_points.shape[1], src_image.shape[2]))
        i_x = range(src_image.shape[1])
        i_y = range(src_image.shape[0])
        flatten_interp_points = interp_points.reshape([interp_points.shape[0]*interp_points.shape[1], 2])
        for i_channel in range(src_image.shape[2]):
            interp_image[:, i_channel] = interpolate.interpn((i_y, i_x), src_image[:, :, i_channel],
                                                            flatten_interp_points[:, [1, 0]], method = sample_method,
                                                            bounds_error=False, fill_value=0)
        interp_image = interp_image.reshape((interp_points.shape[0], interp_points.shape[1], src_image.shape[2]))

        return interp_image

    def _trans_kp_2d(kps, center, dx, dy, lt, ratio):
        kp2d_offset = kps[:, :2] - center
        proj_x, proj_y = np.dot(kp2d_offset, dx), np.dot(kp2d_offset, dy)
        for idx in range(len(kps)):
            kps[idx, :2] = (dx * proj_x[idx] + dy * proj_y[idx] + lt) * ratio
        return kps


    src_image = cv2.imread(image_path)

    center, lt, rt, rb, lb  = _expand_crop_box(lt, rb, scale)

    #calc rotated box
    radian = angle * np.pi / 180.0
    v_sin, v_cos = math.sin(radian), math.cos(radian)

    rot_matrix = np.array([[v_cos, v_sin],[-v_sin, v_cos]])

    n_corner = (np.dot(rot_matrix, np.array([lt - center, rt - center, rb - center, lb - center]).T).T) + center
    n_lt, n_rt, n_rb, n_lb = n_corner[0], n_corner[1], n_corner[2], n_corner[3]

    lt, rt, rb, lb = calc_obb(np.array([lt, rt, rb, lb, n_lt, n_rt, n_rb, n_lb]))
    lt, rt, rb, lb, dx, dy, ratio = _extend_box(center, lt, rt, rb, lb, crop_size = crop_size)
    s_pts = _get_sample_points(lt, rt, rb, lb, crop_size)
    dst_image = _sample_image(src_image, s_pts)
    kp_2d = _trans_kp_2d(kp_2d, center, dx, dy, lt, ratio)

    return dst_image, kp_2d

def flip_image(src_image, kps, mask=None):
    h, w = src_image.shape[0], src_image.shape[1]
    src_image = cv2.flip(src_image, 1)

    kps[:, 0] = w - 1 - kps[:, 0]
    kp_map = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13]
    kps[:, :] = kps[kp_map]
    if mask is None:
        return src_image, kps
    mask = cv2.flip(mask, 1)
    return src_image, kps, mask

# Visualization func.

def draw_lsp_14kp__bone(src_image, pts):
        bones = [
            [0, 1, 255, 0, 0],
            [1, 2, 255, 0, 0],
            [2, 12, 255, 0, 0],
            [3, 12, 0, 0, 255],
            [3, 4, 0, 0, 255],
            [4, 5, 0, 0, 255],
            [12, 9, 0, 0, 255],
            [9,10, 0, 0, 255],
            [10,11, 0, 0, 255],
            [12, 8, 255, 0, 0],
            [8,7, 255, 0, 0],
            [7,6, 255, 0, 0],
            [12, 13, 0, 255, 0]
        ]

        for pt in pts:
            src_image = cv2.circle(src_image,(int(pt[0]), int(pt[1])),2,(0,255,255),-1)
        if pts.shape[0]!=14:
            return src_image
        for line in bones:
            pa = pts[line[0]]
            pb = pts[line[1]]
            if (pa>0).all() and (pb>0).all():
                xa,ya,xb,yb = int(pa[0]),int(pa[1]),int(pb[0]),int(pb[1])
                src_image = cv2.line(src_image,(xa,ya),(xb,yb),(line[2], line[3], line[4]),2)
        return src_image

def plot_mesh(vertices, triangles, subplot = [1,1,1], title = 'mesh', el = 90, az = -90, lwdt=.1, dist = 6, color = "blue"):
    '''
    plot the mesh
    Args:
        vertices: [nver, 3]
        triangles: [ntri, 3]
    '''
    ax = plt.subplot(subplot[0], subplot[1], subplot[2], projection = '3d')
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles = triangles, lw = lwdt, color = color, alpha = 1)
    ax.axis("off")
    ax.view_init(elev = el, azim = az)
    ax.dist = dist
    plt.title(title)
    return plt

def plot_3d_points(points, color = 'r', save_path='test.png'):

    x, y, z = points[:,0], points[:,1],points[:,2]
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c=color)

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig(save_path)

def plot_3d_points_set(points_set, colors = ['r'], save_path='test.png'):
    ax = plt.subplot(111, projection='3d')
    for points,color in zip(points_set,colors):
        x, y, z = points[:,0], points[:,1],points[:,2]
        ax.scatter(x, y, z, c=color)

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.savefig(save_path)

def show3Dpose(kp3ds, lcolor=["#3498db"], rcolor=["#e74c3c"], save_path='test.png',skeleton_type='lsp'): # blue, orange
  """
  Visualize a 3d skeleton
  Args
    kp3d: kp_num x 3 vector.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  #I   = np.array([1,2,3,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  #J   = np.array([2,3,4,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  #LR  = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)#1-left 0-right
  if skeleton_type=='lsp':
    I   = np.array([0,1,2,  5,4,3,  6,7,8,  11,10, 9,  12]) # start points
    J   = np.array([1,2,12, 4,3,12, 7,8,12, 10, 9,12,  13]) # end points
    LR  = np.array([0,0,0,  1,1,1,  0,0,0,   1, 1, 1,   0], dtype=bool)#1-left 0-right
  elif skeleton_type=='smpl':
    I   = np.array([0,0,1,  2,4,5,  0, 12,12,  12,16,17,  18,19]) # start points
    J   = np.array([1,2,4,  5,7,8,  12,15,16,  17,18,19,  20,21]) # end points
    LR  = np.array([1,0,1,  0,1,0,  0,  0, 1,   0, 1, 0,   1, 0], dtype=bool)#1-left 0-right

  for idx,kp3d in enumerate(kp3ds):
      ax = plt.subplot(1,len(kp3ds),idx+1, projection='3d')
      for i in np.arange( len(I) ):
          x, y, z = [np.array( [kp3d[I[i], j], kp3d[J[i], j]] ) for j in range(3)]
          ax.plot(z, x, -y, lw=2, c=lcolor[idx] if LR[i] else rcolor[idx])

          RADIUS = 1 # space around the subject
          xroot, yroot, zroot = 0,0,0#(kp3d[2,0]+kp3d[3,0], kp3d[0,1], kp3d[0,2]
          ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
          ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
          ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

          ax.set_xlabel("x")
          ax.set_ylabel("y")
          ax.set_zlabel("z")

          # Get rid of the ticks and tick labels
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_zticks([])

          ax.get_xaxis().set_ticklabels([])
          ax.get_yaxis().set_ticklabels([])
          ax.set_zticklabels([])
          ax.set_aspect('equal')

          # Get rid of the panes (actually, make them white)
          white = (1.0, 1.0, 1.0, 0.0)
          ax.w_xaxis.set_pane_color(white)
          ax.w_yaxis.set_pane_color(white)
          # Keep z pane

          # Get rid of the lines in 3d
          ax.w_xaxis.line.set_color(white)
          ax.w_yaxis.line.set_color(white)
          ax.w_zaxis.line.set_color(white)

  plt.savefig(save_path)

def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
  """
  Visualize a 2d skeleton
  Args
    channels: 64x1 vector. The pose to plot.
    ax: matplotlib axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  assert channels.size == len(data_utils.H36M_NAMES)*2, "channels should have 64 entries, it has %d instead" % channels.size
  vals = np.reshape( channels, (len(data_utils.H36M_NAMES), -1) )

  I  = np.array([1,2,3,1,7,8,1, 13,14,14,18,19,14,26,27])-1 # start points
  J  = np.array([2,3,4,7,8,9,13,14,16,18,19,20,26,27,28])-1 # end points
  LR = np.array([1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(2)]
    ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

  # Get rid of the ticks
  ax.set_xticks([])
  ax.set_yticks([])

  # Get rid of tick labels
  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])

  RADIUS = 350 # space around the subject
  xroot, yroot = vals[0,0], vals[0,1]
  ax.set_xlim([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim([-RADIUS+yroot, RADIUS+yroot])
  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")

  ax.set_aspect('equal')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf


def line_intersect(sa, sb):
    al, ar, bl, br = sa[0], sa[1], sb[0], sb[1]
    assert al <= ar and bl <= br
    if al >= br or bl >= ar:
        return False
    return True

'''
    return whether two rectangle intersect
    ra, rb left_top point, right_bottom point
'''
def rectangle_intersect(ra, rb):
    ax = [ra[0][0], ra[1][0]]
    ay = [ra[0][1], ra[1][1]]

    bx = [rb[0][0], rb[1][0]]
    by = [rb[0][1], rb[1][1]]

    return line_intersect(ax, bx) and line_intersect(ay, by)

def get_intersected_rectangle(lt0, rb0, lt1, rb1):
    if not rectangle_intersect([lt0, rb0], [lt1, rb1]):
        return None, None

    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = max(lt[0], lt1[0])
    lt[1] = max(lt[1], lt1[1])

    rb[0] = min(rb[0], rb1[0])
    rb[1] = min(rb[1], rb1[1])
    return lt, rb

def get_union_rectangle(lt0, rb0, lt1, rb1):
    lt = lt0.copy()
    rb = rb0.copy()

    lt[0] = min(lt[0], lt1[0])
    lt[1] = min(lt[1], lt1[1])

    rb[0] = max(rb[0], rb1[0])
    rb[1] = max(rb[1], rb1[1])
    return lt, rb

def get_rectangle_area(lt, rb):
    return (rb[0] - lt[0]) * (rb[1] - lt[1])

def get_rectangle_intersect_ratio(lt0, rb0, lt1, rb1):
    (lt0, rb0), (lt1, rb1) = get_intersected_rectangle(lt0, rb0, lt1, rb1), get_union_rectangle(lt0, rb0, lt1, rb1)

    if lt0 is None:
        return 0.0
    else:
        return 1.0 * get_rectangle_area(lt0, rb0) / get_rectangle_area(lt1, rb1)

def convert_image_by_pixformat_normalize(src_image, pix_format, normalize):
    if pix_format == 'NCHW':
        src_image = src_image.transpose((2, 0, 1))

    if normalize:
        src_image = (src_image.astype(np.float) / 255) * 2.0 - 1.0

    return src_image

def align_by_root(joints):
    root_id = 0
    pelvis = joints[:, root_id, :]
    return joints - torch.unsqueeze(pelvis, dim=1)
'''
    align ty pelvis
    joints: n x 14 x 3, by lsp order
'''
def align_by_pelvis(joints):
    left_id = 3
    right_id = 2
    pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.0
    return joints - torch.unsqueeze(pelvis, dim=1)

def align_by_pelvis_single(joints, get_pelvis=False):
    """
    Assumes joints is 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    left_id = 3
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.
    if get_pelvis:
        return joints - np.expand_dims(pelvis, axis=0), pelvis
    else:
        return joints - np.expand_dims(pelvis, axis=0)

def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

# IO functions

def save_pkl(info,name='../data/info.pkl'):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-4:] !='.pkl':
        name += '.pkl'
    with open(name,'wb') as outfile:
        pickle.dump(info, outfile, pickle.HIGHEST_PROTOCOL)
def read_pkl(name = '../data/info.pkl'):
    with open(name,'rb') as f:
        return pickle.load(f)
def read_pkl_coding(name = '../data/info.pkl'):
    with open(name, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
    return p
def check_file_and_remake(path,remove=False):
    if remove:
        if os.path.isdir(path):
            shutil.rmtree(path)
    if not os.path.isdir(path):
        os.makedirs(path)

def save_h5(info,name):
    check_file_and_remake(name.replace(os.path.basename(name),''))
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'w')
    for item, value in info.items():
        f[item] = value
    f.close()

def read_h5(name):
    if name[-3:] !='.h5':
        name += '.h5'
    f=h5py.File(name,'r')
    info = {}
    for item, value in f.items():
        info[item] = np.array(value)
    f.close()
    return info

def h36m32_2_lsp14(h36m32):
    relation = [3,2,1,6,7,8,27,26,25,17,18,19,13,15]
    lsp14 = h36m32[:,relation,:]
    return lsp14
