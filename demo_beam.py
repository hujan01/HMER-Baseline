'''
@Descripttion: 集束搜索
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2020-02-19 16:51:37
@LastEditTime: 2020-07-05 00:08:12
'''

import math 
import os 
import operator
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
from queue import PriorityQueue

from dataset import MERData
from model import Encoder, Decoder
from config import cfg

torch.backends.cudnn.benchmark = False

ignoreTxt = "validIgnore.txt"
ignoreLs = []
with open(ignoreTxt) as fr:
    for l in fr.readlines():
        ignoreLs.append(l.strip())

class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

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

encoder.load_state_dict(torch.load('checkpoints/encoder_47p53.pkl'))
decoder.load_state_dict(torch.load('checkpoints/attn_decoder_47p53.pkl'))

encoder.eval()
decoder.eval()

# 开始推断
fw = open('result_47p53_beam.txt', 'w')
testPath = "/home/hj/workspace/HMER/data/CROHME2016/valid"
maxlen = 70

imgFiles = os.listdir(testPath)
for imgName in imgFiles:
    if os.path.splitext(imgName)[0]  in ignoreLs:
        continue
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

    beam_width = 10
    topk = 2

    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1) # 初始节点
    nodes = PriorityQueue() # 队列

    nodes.put((-node.eval(), node))
    qsize = 1 #队列长度

    while True:
        if qsize > 2000:
            break

        score, n = nodes.get()
        decoder_input = n.wordid
        decoder_hidden = n.h

        if n.wordid.item() == 0 and n.prevNode != None:
            endnodes.append((score, n)) # 保存了所有路径
            # 这里是否要输出多个句子
            if len(endnodes) >= number_required:
                break
            else:
                continue
            
        decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, low_feature, high_feature)
        output = F.log_softmax(decoder_output)
        log_prob, indexes = torch.topk(output, beam_width) # 返回元组

        nextnodes = []
        for new_k in range(beam_width):
            decoder_input = indexes[0][new_k].view(1, 1)
            log_p = log_prob[0][new_k]

            node = BeamSearchNode(decoder_hidden, n, decoder_input, n.logp+log_p, n.leng+1)
            score = -node.eval()
            nextnodes.append((score, node))
        
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))

        qsize += len(nextnodes)-1

    utterances = []
    for score, n in sorted(endnodes, key=operator.itemgetter(0)):
        utterance = []
        utterance.append(n.wordid)
        while n.prevNode != None:
            n = n.prevNode
            utterance.append(n.wordid)
        utterance = utterance[::-1] #倒置
        utterances.append(utterance)

    #打印识别结果
    pred_latex = []
    for j in range(1, maxlen):
        # if int(pred[j]) == 0:
        if int(utterances[0][j]) == 0:
            break 
        else:
            pred_latex.append(worddicts_r[int(utterances[0][j])])
    latex = " ".join(pred_latex)
    imgName = os.path.splitext(imgName)[0]
    fw.write(imgName+'\t')
    fw.write(latex)
    fw.write('\n')
    print(pred_latex)    
fw.close()        

 