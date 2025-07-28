from __future__ import print_function, division
import sys
# sys.path.append('core')
import math
from typing import Dict, Tuple
from pathlib import Path
import weakref
from skimage.transform import rotate, warp
import h5py
import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import imageio
import hdf5plugin
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'


import numpy as np
import torch 

from utils import load_disp_png

# 实现的功能包括：使用DSEC数据集对双目图像进行视差估计

if __name__ == '__main__':

    # DSEC: RGB 参数
    baseline = 0.51  # 单位为m
    focal_length = 1150.8943600390282
    cx = 723.4334411621094
    cy = 572.102180480957

    # name = 'zurich_city_11_c'
    dir = 'F:/Research/Experiment_Code/2024CVPR/Dataset/Test_DSEC/zurich_city_05_b/'
    
    # 读取dsip并转换为深度
    disp_gt, mask_gt= load_disp_png(dir + "disparity/image/000000.png")
    disp_gt[disp_gt < 0] = 0
    depth_gt = (focal_length * baseline) / (disp_gt + 1e-6)
    depth_gt[depth_gt > 40] = 0
    
    
    # 读取左右视图的图像
    left_img = cv2.imread(dir + 'images/000000.png')
    right_img = cv2.imread(dir + 'images_right/000000.png')

    # 转换为灰度图像
    # left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    # right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 创建 SGBM 视差估计对象
    window_size = 8
    min_disp = 1
    num_disp = 64
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=-1,
        preFilterCap=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=100,
        mode=cv2.STEREO_SGBM_MODE_HH
    )

    # 计算视差图
    disparity = stereo.compute(left_img, right_img)
    disparity = np.divide(disparity.astype(np.float32), 16.)#除以16得到真实视差图

    # 视差图归一化
    # disparity = disparity - min_disp
    disparity[disparity < 0] = 0

    
    
    depth_map = (focal_length * baseline) / (disparity + 1e-6)

    depth_map[depth_map>40.0] = 0
    
    depth_map /= 256.0
    # disparity = (disparity - min_disp) / max_disp
    # print(depth_map)
    print(depth_gt)
    print(depth_map)
    # depth_map = depth_map / np.max(depth_map)
    # for j in range(disp_gt.shape[0]):
    #     for i in range(disp_gt.shape[1]):
    #         if disp_gt[j, i] != 0:
    #             print("position: %d, %d, disp_gt: %f, disp_pred: %f" % (i,j,depth_gt[j, i], depth_map[j, i]))
                
    

    # 可视化视差图
    cv2.imshow('depth map', depth_map)
    cv2.waitKey(0)
    cv2.destroyAllWindows()