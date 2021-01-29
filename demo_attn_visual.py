'''
@Descripttion: 
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-02-19 16:51:37
LastEditTime: 2020-12-29 13:16:55
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
import torchvision.transforms as transforms

from dataset import MERData
from model import Encoder, Decoder
from config import cfg
from utils.util import get_all_dist, load_dict, custom_dset, collate_fn_double, show_attention_images

torch.backends.cudnn.benchmark = False

# 配置参数
valid_datasets = ['data/test.pkl', 'data/test2016.txt']
dictionaries = 'data/dictionary.txt'
demo_path = "data/demo"
img_name = "23_em_64.bmp"

Imagesize = 500000
bs = 1
maxlen = 70
maxImagesize = 100000
hidden_size = 256

worddicts = load_dict(dictionaries)  #token2id
worddicts_r = [None] * len(worddicts)   #id2token
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk 

# 1.1加载图片
img_path = os.path.join(demo_path, img_name)
img = cv2.imread(img_path)

# 1.2图片处理
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (h, w)
pil_img = Image.fromarray(img) # (w, h)
img = torch.from_numpy(img).unsqueeze(0).float()

img_h, img_w = img.size()[1:]
img_mask = 255.0*torch.ones(1, img_h, img_w).type(torch.FloatTensor)
img = torch.cat((img, img_mask), dim=0)
img = (img.unsqueeze(0)/255.0).cuda()

# 2.加载模型
encoder = Encoder(img_channels=2)
decoder = Decoder(112)

encoder = encoder.cuda()
decoder = decoder.cuda()

encoder.load_state_dict(torch.load('checkpoints/encoder_47p21.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_47p21.pkl'))

encoder.eval()
decoder.eval()

# 3.开始识别
print('recognize {} ....'.format(img_name))

# 编码器
feature = encoder(img)

# 解码器初始化
decoder_input = torch.LongTensor([111]).view(1, 1).cuda()
decoder_hidden = decoder.init_hidden(1).cuda()
decoder.reset(bs, feature.size())

pred = torch.zeros(maxlen)

for i in range(maxlen):
    decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, feature)
    
    topv, topi = torch.max(decoder_output, 1) 
    if torch.sum(topi)==0:
        break
    
    decoder_input = topi
    decoder_input = decoder_input.view(bs, 1)
    
    # prediction
    pred[i] = decoder_input

pred_latex = []
for j in range(maxlen):
    if int(pred[j]) == 0:
        break 
    else:
        pred_latex.append(worddicts_r[int(pred[j])])

attn = decoder.coverage_attn.alpha
show_attention_images(pil_img, pred_latex, attn, feature.size(2), feature.size(3), smooth=True)

print(pred_latex)  