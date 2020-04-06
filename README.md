# Human Mesh Recovery from Monocular Images via a Skeleton-disentangled Representation

[Sun Yu](https://scholar.google.com/citations?hl=en&user=fkGxgrsAAAAJ), [Ye Yun](https://scholar.google.com/citations?hl=en&user=wxvX51gAAAAJ), [Liu Wu](https://scholar.google.com/citations?hl=en&user=rQpizr0AAAAJ), [Gao Wenpeng](http://homepage.hit.edu.cn/wpgao), [Fu YiLi](http://homepage.hit.edu.cn/fuyili), [Mei Tao](https://scholar.google.com/citations?user=7Yq4wf4AAAAJ)

Accepted to ICCV 2019 https://arxiv.org/abs/1908.07172 [Paper Link](https://arxiv.org/abs/1908.07172)

![Demo Image](https://github.com/Arthur151/DSD-SATN/blob/master/resources/results/video_result.png)

### Internet Video Demo
![Demo of internet skating video](https://github.com/Arthur151/DSD-SATN/blob/master/resources/video/skate3.gif)

### More Demos on YouTube
[![Demo video on Youtube](http://i3.ytimg.com/vi/GG-8If4uVQM/maxresdefault.jpg)](https://youtu.be/GG-8If4uVQM=640x360)

### Requirements
- Python 3.6+
- [Pytorch](https://pytorch.org/) tested on 0.4.1/1.0/1.2 versions
- [PyTorch implementation of the Neural 3D Mesh Renderer](https://github.com/daniilidis-group/neural_renderer) for visualization

### Installation
```
pip install -r requirements.txt
```

### Demo

Simply go into DSD-SATN/src/, and run
```
sh run.sh
```
The results are saved in DSD-SATN/resources/results.

![Demo Results](https://github.com/Arthur151/DSD-SATN/blob/master/resources/results/im0002.jpg)
![Demo Results](https://github.com/Arthur151/DSD-SATN/blob/master/resources/results/im0028.jpg)
![Demo Results](https://github.com/Arthur151/DSD-SATN/blob/master/resources/results/im0069.jpg)
![Demo Results](https://github.com/Arthur151/DSD-SATN/blob/master/resources/results/im0153.jpg)

### Re-implementation

1. Prepare model and data.

Step 1) Download the pre-trained models and statistical model from [google drive](https://drive.google.com/open?id=1lwqCg7AmAN6hklWzWgB1FhLNBDkECdct). Unzip them under the project dir (e.g. DSD-SATN/trained_model, DSD-SATN/model)

Step 2) Download the [processed annotations](https://drive.google.com/open?id=1-SbuyxPduh1drB0BmDZEYsJNlgnRdGgh) (moshed parameters, extracted DSD features of images in Human3.6M, and 2D pose estimations from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose/tree/pytorch)) of Human3.6M and 3DPW dataset. Unzip them and set their location in data_set_path of src/config.py like
```
data_set_path = {
    'h36m':'PATH/TO/H36M',
    ...
    'pw3d':'/PATH/TO/3DPW',}
```

Step 3) Apply for the datasets from official [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/). Especially, pre-processing the input images of Human3.6M dataset. Extracting 1 frame from every 5 frame of video in Human3.6M dataset. Set the path of extracted images as {H36m_dir}/images and name each image as the format (Sn_action name_camera id_frame number.jpg) shown in h36m/h36m_test.txt (e.g. S11_Discussion 2_2_149.jpg). 

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

### Saving results & Visualization

Additionally, if you want to save the results, please add some options:
```
--save-obj # saving obj file of 3D human body mesh.
--save-smpl-params # saving smpl parameters.
--visual-all # saving all visual rendering results.
```
For example, saving all results of single-frame DSD network on Human3.6M dataset, just type
```
CUDA_VISIBLE_DEVICES=0 python3 test.py --gpu=0 --dataset=h36m --tab=single_h36m --eval --save-obj --save-smpl-params --visual-all
```

### TODO List
- [ ] Releasing code for
    - [x] testing
    - [x] demo of single image
    - [ ] webcam demo

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
