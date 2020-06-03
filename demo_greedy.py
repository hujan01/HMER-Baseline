'''
@Descripttion: 推断
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-02-19 16:51:37
@LastEditTime: 2020-06-02 23:04:53
'''

import math 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from matplotlib import pyplot as plt
from matplotlib import cm
import skimage
from skimage import transform

from dataset import MERData
from model import Encoder, Decoder
from config import cfg

torch.backends.cudnn.benchmark = False

def cmp_result(label, rec):
    dist_mat = np.zeros((len(label)+1, len(rec)+1), dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label), hit_score, ins_score, del_score

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

dictionaries = 'data/dictionary.txt'  
worddicts = load_dict(dictionaries)  #token 2 id
worddicts_r = [None] * len(worddicts)   #id 2 token
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk 

# 加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(112)

encoder = encoder.cuda()
decoder = decoder.cuda()

encoder.load_state_dict(torch.load('checkpoints/encoder_36p93.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_36p93.pkl'))

encoder.eval()
decoder.eval()

# 开始推断
fw = open('result/result_.txt', 'w')

testPath = "/home/hj/workspace/HMER/data/CROHME2016/valid"
maxlen = 70

imgFiles = os.listdir(testPath)
for imgName in imgFiles:
    imgPath = os.path.join(testPath, imgName)
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = torch.from_numpy(img).unsqueeze(0).float()
    img_h, img_w = img.size()[1:]
    img_mask = 255.0*torch.ones(1, img_h, img_w).type(torch.FloatTensor)
    img = torch.cat((img, img_mask), dim=0)
    
    print('recognize {} ....'.format(imgName))
    img = (img.unsqueeze(0)/255.0).cuda()
    low_feature, high_feature = encoder(img)

    # 初始化输入:
    decoder_input = torch.LongTensor([111]).view(1, 1).cuda()
    decoder_hidden = decoder.init_hidden(1).cuda()
    decoder.reset(1, low_feature.size(), high_feature.size())

    pred = torch.zeros(maxlen)
    img_show = cv2.imread(imgPath, 0)
    h, w = img_show.shape[:]
    # plt.subplot(211)
    # plt.imshow(img_show)
    for i in range(maxlen):
        # decoder_output, decoder_hidden, low_alpha, high_alpha = decoder(decoder_input, decoder_hidden, low_feature, high_feature)
        decoder_output, decoder_hidden, low_alpha = decoder(decoder_input, decoder_hidden, low_feature, high_feature)
        output = F.log_softmax(decoder_output)
        print(torch.sum(output, dim=1))
        topv, topi = torch.max(output, 1)
        if torch.sum(topi) == 0:
            break
        decoder_input = topi.view(1, 1)
        pred[i] = decoder_input.flatten()
        
        # low_alpha = low_alpha[0, 0, :, :].cpu().detach().numpy()
        # low_pos = np.unravel_index(np.argmax(low_alpha), low_alpha.shape)
        # high_alpha = high_alpha[0, 0, :, :].cpu().detach().numpy()
        # high_pos = np.unravel_index(np.argmax(high_alpha), high_alpha.shape)
        # low_attn_img = np.zeros(low_alpha.shape)
        # high_attn_img = np.zeros(high_alpha.shape)
        # low_attn_img[low_pos[0], low_pos[1]] = 255
        # high_attn_img[high_pos[0], high_pos[1]] = 255
        # # print(low_alpha[pos[0], pos[1]])
        # low_alpha_img = transform.pyramid_expand(low_attn_img, upscale=16, sigma=2)
        # high_alpha_img = transform.pyramid_expand(high_attn_img, upscale=8, sigma=2)
        # low_pad_h = max(low_alpha_img.shape[0], h)-min(low_alpha_img.shape[0], h)
        # low_pad_w = max(low_alpha_img.shape[1], w)-min(low_alpha_img.shape[1], w)
        # high_pad_h = max(high_alpha_img.shape[0], h)-min(high_alpha_img.shape[0], h)
        # high_pad_w = max(high_alpha_img.shape[1], w)-min(high_alpha_img.shape[1], w)
        # low = np.pad(low_alpha_img, ((low_pad_h, 0), (low_pad_w, 0)), 'constant', constant_values=(0,0))
        # high = np.pad(high_alpha_img, ((high_pad_h, 0), (high_pad_w, 0)), 'constant', constant_values=(0,0))
        # attn = low+high
        # attn_img = cv2.addWeighted(img_show,0.5,attn.astype(np.uint8),0.5,0)
        # cv2.imwrite('result/{}.png'.format(str(i)), attn_img)
        # print(worddicts_r[int(pred[i])])

    # 打印识别结果
    pred_latex = []
    for j in range(maxlen):
        if int(pred[j]) == 0:
            break 
        else:
            pred_latex.append(worddicts_r[int(pred[j])])
    latex = " ".join(pred_latex)
    imgName = os.path.splitext(imgName)[0]
    fw.write(imgName+'\t')
    fw.write(latex)
    fw.write('\n')
    print(pred_latex)    
fw.close()        

 