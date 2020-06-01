'''
@Descripttion:
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-02-19 16:51:37
@LastEditTime: 2020-04-22 15:00:59
'''

import math
import os 

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data

from dataset import MERData
from model import Encoder, Decoder
from config import cfg

torch.backends.cudnn.benchmark = False
valid_datasets = ['test.pkl', 'data/test2016.txt']
dictionaries = 'data/dictionary.txt'
Imagesize = 500000
batch_size_t = 1
maxlen = 70
maxImagesize = 200000
hidden_size = 256
gpu = [0]

def cmp_result(label, rec):
    # 动态规划计算编辑距离
    dist_mat = np.zeros((len(label)+1, len(rec)+1), dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            sub_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1]) #替换， 相同加0，不同加1
            ins_score = dist_mat[i,j-1] + 1 #插入
            del_score = dist_mat[i-1, j] + 1 #删除
            dist_mat[i,j] = min(sub_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label), sub_score, ins_score, del_score

def load_dict(dictFile):
    """ 加载字典 """
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon [w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon
    
worddicts = load_dict(dictionaries)  #token 2 id
worddicts_r = [None] * len(worddicts)   #id 2 token
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk 

# 返回所有测试图片和标签  (c,h,w) (latex字符)
test, test_label, uidList = MERData(
                                valid_datasets[0], valid_datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=Imagesize, maxlen=maxlen, maxImagesize=maxImagesize
                          )

class custom_dset(data.Dataset):
    """ 增加一个图片名输出 """
    def __init__(self, train, train_label, uidList):
        self.train = train
        self.train_label = train_label
        self.uidList = uidList
    def __getitem__(self, index):
        train_setting = torch.from_numpy(np.array(self.train[index]))
        label_setting = torch.from_numpy(np.array(self.train_label[index])).type(torch.LongTensor)
        uid_setting = self.uidList[index]

        size = train_setting.size()
        train_setting = train_setting.view(1, size[2], size[3])
        label_setting = label_setting.view(-1)

        return train_setting, label_setting, uid_setting

    def __len__(self):
        return len(self.train)

image_test = custom_dset(test, test_label, uidList)

def collate_fn(batch):
    """ 引入掩码 
        加入了图片名输出
    """
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label, uid = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    return img_padding_mask, label_padding, uid[0]

test_loader = torch.utils.data.DataLoader(
    dataset = image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = 0,
)

# 1. 加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(112)

encoder = encoder.cuda()
decoder = decoder.cuda()

encoder.load_state_dict(torch.load('checkpoints/encoder_36p93.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_36p93.pkl'))

encoder.eval()
decoder.eval()

# eval params
total_dist = 0 # 统计所有的序列的总编辑距离
total_label = 0 # 统计所有序列的总长度
total_line = 0 # 统计一共有多少个序列
total_line_rec = 0 # 统计识别正确的序列
error1, error2, error3 = 0, 0, 0

def parse(label_sub, prediction_sub, f):

    label_sub = [x for x in label_sub if x != 0]
    prediction_sub = [x for x in prediction_sub if x!=0]
    if label_sub != prediction_sub:
        f.write(str(len(label_sub)))
        f.write('\n')

# 2. 开始评估
f = open('result.txt', 'w')

for step_t, (x_t, y_t, uid) in enumerate(test_loader): 

    x_t = x_t.cuda()
    y_t = y_t.cuda()
    low_feature_t, high_feature_t = encoder(x_t)

    # 初始化输入
    decoder_input_t = torch.LongTensor([111]*batch_size_t).view(-1, 1).cuda()
    decoder_hidden_t = decoder.init_hidden(batch_size_t).cuda()
    decoder.reset(batch_size_t, low_feature_t.size(), high_feature_t.size()) # 每个时间步需要重置

    prediction = torch.zeros(batch_size_t, maxlen)

    prediction_sub = []
    prediction_real = []    

    label_sub = []
    label_real = []

    # 处理标签
    m = torch.nn.ZeroPad2d((0, maxlen-y_t.size()[1], 0, 0))
    y_t = m(y_t)

    for i in range(maxlen):
        decoder_output_t, decoder_hidden_t, _ = decoder(decoder_input_t, decoder_hidden_t, low_feature_t, high_feature_t)
        
        topv, topi = torch.max(decoder_output_t, 1) 
        if torch.sum(topi)==0: # 一个bs中所有序列都预测结束
            break
        
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t, 1)
        
        # prediction
        prediction[:, i] = decoder_input_t

    for i in range(batch_size_t):
        for j in range(maxlen):
            if int(prediction[i][j]) == 0:
                break
            else:
                prediction_sub.append(int(prediction[i][j]))
                prediction_real.append(worddicts_r[int(prediction[i][j])])

        if len(prediction_sub) < maxlen: #不足后面填0
            prediction_sub.append(0)

        for k in range(y_t.size()[1]):
            if int(y_t[i][k]) == 0:
                break
            else:
                label_sub.append(int(y_t[i][k]))
                label_real.append(worddicts_r[int(y_t[i][k])])
        label_sub.append(0)

        # 评价指标
        dist, llen, sub, ins, dls = cmp_result(label_sub, prediction_sub)
        # parse(label_sub, prediction_sub, f)     
        # print(dist, llen, sub, ins, dls)
        wer_step = float(dist) / llen

        total_dist += dist
        total_label += llen
        total_line += 1 

        if dist == 0:
            total_line_rec = total_line_rec + 1
        if dist == 1:
            error1 += 1
        if dist == 2:
            error2 += 1
        if dist == 3:
            error3 += 1
    
        print('step is %d' % (step_t))
        print('prediction is ')
        print(prediction_real)
        print('the truth is')
        print(label_real)
        print('the wer is %.5f' % (wer_step))
        
        #将预测结果写入到文件中
        f.write(uid+'\t')
        f.write(' '.join(prediction_real)+'\n')

        label_sub = []
        prediction_sub = []
        label_real = []
        prediction_real = []
        
f.close()

wer = float(total_dist) / total_label
exprate = float(total_line_rec) / total_line
error1 += total_line_rec
error2 += error1
error3 += error2

# avg wer and ExpRate
print('{}/{}'.format(total_line_rec, total_line))
print('ExpRate is {:.4f}'.format(exprate))

print('error1 nums: {}, error2 nums: {}, error3 nums: {}'.format(error1, error2, error3))
print('error1 is {:.4f}, error2 is {:.4f}, error3 is {:.4f}'.format((error1)/total_line, error2/total_line, error3/total_line))

print('wer is {:.4f}'.format(wer))




  


