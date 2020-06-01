'''
@Descripttion: 预生成二进制文件
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-04-21 19:13:08
@LastEditTime: 2020-04-21 21:43:16
'''
import os
import sys

import pandas as pd
import pickle as pkl
import numpy as np
import cv2
from imageio import imread, imsave

image_path = "/home/hj/workspace/HMER/data/CROHME2016/test/"

channels=1
sentNum=0
features={}

scpFile = open('/home/hj/workspace/HMER/data/test2016.txt')
lines = scpFile.readlines()
for l in lines:
    line = l.strip() # remove the '\r\n'
    if not line: 
        break
    else:
        key = line.split('\t')[0]
        image_file = image_path + key + '.bmp'
        im = cv2.imread(image_file)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) 

        mat = np.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8') # (1, h, w)
        
        for channel in range(channels):
            image_file = image_path + key + '.bmp'
            im = cv2.imread(image_file)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            mat[channel, :, :] = im
            
        sentNum = sentNum + 1
        features[key] = mat

        print('process sentences ', sentNum)

print('load images done. sentence number ', sentNum)

outFile = 'test.pkl'
oupFp_feature = open(outFile, 'wb')
pkl.dump(features, oupFp_feature)

print('save file done')

