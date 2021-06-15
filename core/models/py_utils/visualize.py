# -*- coding: utf-8 -*-
import numpy as np
import cv2
import torch
import random

def visualize(image, tl_heat, br_heat):
    # image'size = [2, 3, 1151, 2047]
    # 这个图片的原始尺寸下采样4倍，就变成heatmap的维度，所以测试时不一定是128
    # [2, 7, 288, 512]，这个是不定的，看图片大小
    tl_heat = torch.sigmoid(tl_heat)
    # [2, 7, 288, 512]
    br_heat = torch.sigmoid(br_heat)
    
    
    # 这个colors是一个list，shape为(7, 1, 1, 3)，7是类别数，1，1，3是随机random的
    # 这个作用就是给每个类定制了专属的随机生成的颜色,大概长下面这样
    '''
    [array([[[105, 131, 151]]], dtype=uint8),
     array([[[180, 216, 153]]], dtype=uint8),
     array([[[151, 150, 167]]], dtype=uint8),
     array([[[188, 236, 177]]], dtype=uint8),
     array([[[111, 143, 220]]], dtype=uint8),
     array([[[240, 194, 238]]], dtype=uint8),
     array([[[207, 136, 124]]], dtype=uint8)]
    '''
    colors = [[0, 0, 255],
            [0, 255, 0],
            [255, 0, 0],
            [255, 140, 0],
            [255, 255, 0]]
    print(colors)
    # tl_heat[0] size = [7, 288, 512]
    # 取走第一个batch的特征，配上颜色
    # tl_hm、br_hm的维度均是[h, w, 3]
    tl_hm = _gen_colormap(tl_heat[0].detach().cpu().numpy(), colors)
    br_hm = _gen_colormap(br_heat[0].detach().cpu().numpy(), colors)
    # 标准差和均值
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                    dtype=np.float32).reshape(3, 1, 1)
    # 为rgb的图片，每通道乘上标准差加上均值，相当于每通道分配一个数字
    img = (image[0].detach().cpu().numpy() * std + mean) * 255
    # 再把图片transpose成标准的样子
    img = img.astype(np.uint8).transpose(1, 2, 0)

    tl_blend = _blend_img(img, tl_hm)
    br_blend = _blend_img(img, br_hm)
    count = random.randint(0,10000)
    cv2.imwrite("/home/jhsu/yjl/lite/demo_result/tl_heatmap_" + str(count) + ".jpg", tl_blend)
    cv2.imwrite("/home/jhsu/yjl/lite/demo_result/br_heatmap_" + str(count) + ".jpg", br_blend)
    print("~~~save heatmaps OK!")

def _gen_colormap(heatmap, colors):
    # 这个heatmap的维度是[7, 288, 512]
    num_classes = heatmap.shape[0]
    h, w = heatmap.shape[1], heatmap.shape[2]
    color_map = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        # np.maximum是两个输入进行对比，每次谁大就挑谁的，维度要一致
        # color_map维度[h, w, 3]
        # heatmap[i, :, :, np.newaxis]维度[h, w, 1]
        # colors[i]维度[1, 1, 3]
        # 最终右边这一长串其实是0-255的整型数字
        # 接着循环类别次，color_map一直更新，每次挑maximum的
      color_map = np.maximum(
        color_map, (heatmap[i, :, :, np.newaxis] * colors[i]).astype(np.uint8))
    return color_map


def _blend_img(back, fore, trans=0.7):
    '''
    back = img-->[h*4, w*4, 3]
    fore = tl_hm-->[h, w, 3]
    '''
    if fore.shape[0] != back.shape[0] or fore.shape[0] != back.shape[1]:
      fore = cv2.resize(fore, (back.shape[1], back.shape[0]))
    if len(fore.shape) == 2:
      fore = fore.reshape(fore.shape[0], fore.shape[1], 1)
    # 两幅图像进行合并时，按公式：blended_img = img1 * (1 – alpha) + img2* alpha 进行
    ret = (back * (1. - trans) + fore * trans).astype(np.uint8)
    # 别越界了,ret的大小就是原图的大小
    ret[ret > 255] = 255
    ret[ret < 0] = 0
    return ret