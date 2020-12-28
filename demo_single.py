'''
@Descripttion: 
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-02-19 16:51:37
LastEditTime: 2020-12-28 11:51:49
'''
import math 
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import image as mpimg
from skimage import transform
from skimage.io import imread

from dataset import MERData
from model import Encoder, Decoder
from config import cfg
from utils.util import cmp_result, load_dict

torch.backends.cudnn.benchmark = False

dictionaries = 'data/dictionary.txt'  
worddicts = load_dict(dictionaries)  #token 2 id
worddicts_r = [None] * len(worddicts)   #id 2 token
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk 

def getWH(img_w, img_h):
    '''
    模仿卷积层 encoder 缩放
    '''
    img_w, img_h = np.ceil(img_w / 4), np.ceil(img_h / 4)
    img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    # img_w, img_h = np.ceil(img_w / 2), np.ceil(img_h / 2)
    img_w, img_h = np.ceil(img_w - 1), np.ceil(img_h - 1)
    return int(img_w), int(img_h)
    
def readImageAndShape(img_path):
    lena = mpimg.imread(img_path)  # 读取目录下的图片，返回 np.array
    img = imread(img_path)
    img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)*255 - img

    # img = greyscale(img)

    img_w, img_h = lena.shape[1], lena.shape[0]
    return img, img_w, img_h        

def getOutArray(attentionVector, att_w, att_h):
    '''
    将 attentionVector 重新构建为宽 att_w, 高 att_h 的图片矩阵
    '''
    att = sorted(list(enumerate(attentionVector[0].flatten())),
                 key=lambda tup: tup[1],
                 reverse=True)  # attention 按权重从大到小递减排序
    
    idxs, att = zip(*att)

    # 这里取前10个
    thres = att[9] # 阈值
    att_sum = sum(att[:10])
    scale = 1/att_sum

    positions = idxs[:]

    # 把扁平化的一维的 attention slice 重整成二维的图片矩阵，像素颜色值范围 [0, 255]
    outarray = np.ones((att_h, att_w)) * 255.

    for i in range(len(positions)):
        pos = positions[i]
        loc_x = int(pos / att_w)
        loc_y = int(pos % att_w)
        att_pos = att[i]
        # 这里设一个阈值过滤一下
        if att_pos > thres :
            att_pos *= scale 
        else:
            att_pos = 0
            
        outarray[loc_x, loc_y] = (1 - att_pos) * 255.
        # (1 - att_pos) * 255. 而不是直接 att_pos * 255
        # 因为颜色值越小越暗，而权重需要越大越暗
    return outarray
    
def getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h):
    outarray = getOutArray(attentionVector, att_w, att_h)

    out_image = Image.fromarray(outarray).resize((img_w, img_h), Image.NEAREST)
    inp_image = Image.open(img_path)
    inp_image = ImageOps.invert(inp_image)
    combine = Image.blend(inp_image.convert('RGBA'), out_image.convert('RGBA'), 0.5)
    return np.asarray(combine)
    
def vis_attention_slice(attentionVector, img_path, path_to_save_attention, img_w, img_h, att_w, att_h):
    combine = getCombineArray(attentionVector, img_path, img_w, img_h, att_w, att_h)
    plt.figure()
    plt.imsave(path_to_save_attention, combine)
    
def getFileNameToSave(path_to_save_attention, i):
    return path_to_save_attention+"_"+str(i)+".png"

def vis_attention_slices(img_path, path_to_save_attention, alpha, i):
    '''
    可视化所有的 attention slices，保存为 png
    '''
    img, img_w, img_h = readImageAndShape(img_path)
    att_w, att_h = getWH(img_w, img_h)

    filename = getFileNameToSave(path_to_save_attention, i)
    vis_attention_slice(alpha, img_path, filename, img_w, img_h, att_w, att_h)
    
def vis_img_with_attention(img_path, dir_output, alpha, i):
    img, img_w, img_h = readImageAndShape(img_path)
    att_w, att_h = getWH(img_w, img_h)
    print("image path: {0} shape: {1}".format(img_path, (img_w, img_h)))

    path_to_save_attention = dir_output+"vis/vis_"+img_path.split('/')[-1][:-4]
    vis_attention_slices(img_path, path_to_save_attention, alpha, i)

# 加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(112)

encoder = encoder.cuda()
decoder = decoder.cuda()

encoder.load_state_dict(torch.load('checkpoints/encoder_47p53.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_47p53.pkl'))

encoder.eval()
decoder.eval()

demoPath = "/home/hj/workspace/HMER/data/demo"
maxlen = 70

imgName = "bg.bmp"

imgPath = os.path.join(demoPath, imgName)
img = cv2.imread(imgPath)

# 图片处理
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

for i in range(maxlen):
    decoder_output, decoder_hidden, high_alpha = decoder(decoder_input, decoder_hidden, low_feature, high_feature)

    output = F.log_softmax(decoder_output)
    print(torch.sum(output, dim=1))
    topv, topi = torch.max(output, 1)
    if torch.sum(topi) == 0:
        break
    decoder_input = topi.view(1, 1)
    pred[i] = decoder_input.flatten()
    
pred_latex = []
for j in range(maxlen):
    if int(pred[j]) == 0:
        break 
    else:
        pred_latex.append(worddicts_r[int(pred[j])])
latex = " ".join(pred_latex)
imgName = os.path.splitext(imgName)[0]

# 打印识别结果
print(pred_latex)    
     

