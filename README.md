# Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation

Sun Yu, Ye Yun, Liu Wu, Gao Wenpeng, Fu YiLi, Mei Tao
ICCV 2019

[Paper Page](https://arxiv.org/abs/1908.07172)
![Teaser Image](https://akanazawa.github.io/hmr/resources/images/teaser.png)

### Requirements
- Python 3.6+
- [Pytorch](https://pytorch.org/) tested on 0.4.1/1.0/1.2 versions
- [PyTorch implementation of the Neural 3D Mesh Renderer](https://github.com/daniilidis-group/neural_renderer) for visualization

### Installation
```
pip install -r requirements.txt
```

### Demo

1. Prepare model and data.

Download the pre-trained models and statistical model from [google drive](https://drive.google.com/open?id=1lwqCg7AmAN6hklWzWgB1FhLNBDkECdct). Unzip them under the project dir (e.g. DSD-SATN/trained_model, DSD-SATN/model)

Apply the image data from official [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/). Download the processed annotations of all datasets from [google drive](https://drive.google.com/open?id=1-SbuyxPduh1drB0BmDZEYsJNlgnRdGgh). Unzip them and set the location of them in data_set_path of src/config.py like
```
data_set_path = {
    'h36m':'PATH/TO/H36M',
    ...
    'mosh':'/PATH/TO/MOSH',}
```

2. Re-implement the evaluation results on Human3.6M and 3DPW datasets.
```
# Evaluating single-frame DSD network on Human3.6M dataset
CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu=0 --dataset=h36m --tab=single_h36m --eval

# Evaluating entire network (DSD-SATN) on Human3.6M dataset
CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu=0 --dataset=h36m --tab=video_h36m --eval --video --eval-with-single-frame-network

# Evaluating single-frame DSD network on 3DPW dataset
CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu=0 --tab=single_3dpw --eval --eval-pw3d

# Evaluating entire network (DSD-SATN) on 3DPW dataset
CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu=0 --tab=video_3dpw --eval --eval-pw3d --video --eval-with-single-frame-network
```
Additionally, if you want to save the mesh results, please add --save-obj for saving obj file of 3D human body mesh and --save-smpl-params for saving smpl parameters. If you want to save all visual rendering results, please add --visual-all.

3. Run the demo

Coming soon...

### TODO List
- [] Releasing code for
    - [x] testing
    - [ ] webcam demo
    - [ ] training

### Citation
If you use this code for your research, please consider citing:
```
@InProceedings{sun2019dsd-satn,
title = {Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation},
author = {Sun, Yu and Ye, Yun and Liu, Wu and Gao, Wenpeng and Fu, YiLi and Mei, Tao},
booktitle = {IEEE International Conference on Computer Vision, ICCV},
year = {2019}
}
```

### Acknowledgement
we refer to [pytorch_hmr](https://github.com/MandyMo/pytorch_HMR) for training code. The fast rendering module is brought from [face3d](https://github.com/YadiraF/face3d). The transformer module is brought from [transformer-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch).

