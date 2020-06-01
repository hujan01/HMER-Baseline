'''
@Descripttion: 数据加载
@Version: 
@Author: jianh
@email: 595495856@qq.com
@Date: 2020-02-19 16:51:51
@LastEditTime: 2020-04-22 14:48:44
'''
import sys

import numpy
import pickle as pkl
import torch

def MERData(feature_file, label_file, dictionary, batch_size, batch_Imagesize, maxlen, maxImagesize): 
    # img
    fp = open(feature_file, 'rb')
    features = pkl.load(fp) # dict
    fp.close()
    # label
    fp2 = open(label_file, 'r')
    labels = fp2.readlines()
    fp2.close()
    
    len_label = len(labels) 

    targets = {}
    # token_to_id
    for l in labels:
        tmp = l.strip().split()
        uid = tmp[0] # image_name
        w_list = [] 
        for w in tmp[1:]: 
            if w in dictionary:
                w_list.append(dictionary[w])
            else:
                print('a word not in the dictionary !! sentence ', uid, 'word ', w)
                sys.exit()
        targets[uid] = w_list

    imageSize={} 
    imagehigh={}
    imagewidth={}
    for uid, fea in features.items():
        imageSize[uid] = fea.shape[1]*fea.shape[2]
        imagehigh[uid] = fea.shape[1]
        imagewidth[uid] = fea.shape[2]

    imageSize = sorted(imageSize.items(), key=lambda d:d[1], reverse=True) #按照h*w排序

    feature_batch = []
    label_batch = []
    feature_total = []
    label_total = []
    uidList = [] # 保存满足条件的图片名

    batch_image_size = 0 
    biggest_image_size = 0 # 数据集图片中的最大尺寸
    i = 0
    # imageSize 根据图片尺寸大小key:label 保存在字典中
    # features uid: feature  targets uid: label
    for uid, size in imageSize:
        if size > biggest_image_size: 
            biggest_image_size = size
        fea = features[uid] 
        lab = targets[uid]
        batch_image_size = biggest_image_size*(i+1)

        # 这里会剔除一些不符合条件的图片
        if len(lab) > maxlen: # 公式最大长度限制
            print('{} label_len: {}'.format(uid, len(lab)))
            continue
        elif size > maxImagesize: # 图像最大长度限制
            print('{} size: {}'.format(uid, size))
            continue
        else:
            uidList.append(uid) 
            if batch_image_size > batch_Imagesize or i == batch_size:
                if label_batch:
                    feature_total.append(feature_batch)
                    label_total.append(label_batch)
                i = 0
                biggest_image_size=size
                feature_batch = []
                label_batch = []
                feature_batch.append(fea)
                label_batch.append(lab)
                batch_image_size = biggest_image_size*(i+1)
                i += 1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i += 1

    # 下面这两个是对应的
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    len_ignore = len_label - len(label_total)
    print('total ', len(label_total), 'batch data loaded')
    print('ignore', len_ignore, 'images')

    return feature_total, label_total, uidList
