'''
@Descripttion: 
@Version: 
@Author: jianh
@Email: 595495856@qq.com
@Date: 2019-12-16 16:00:16
@LastEditTime: 2020-06-03 23:05:46
'''
import math
import random
import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data
from torch import optim

from model import Encoder, Decoder
from dataset import MERData
from config import cfg

# 设置随机数种子
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)

def cmp_result(label, rec):
    """ 编辑距离 """
    dist_mat = np.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def load_dict(dictFile):
    fp = open(dictFile)
    stuff = fp.readlines()
    fp.close()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

exprate = 0 # 是否保存模型 
best_wer = 2**31
# 字典
worddicts = load_dict(cfg.dictionaries)
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

class custom_dset(data.Dataset):
    """ 格式化数据 """
    def __init__(self, train, train_label):
        self.train = train
        self.train_label = train_label

    def __getitem__(self, index):
        train_setting = torch.from_numpy(np.array(self.train[index]))
        label_setting = torch.from_numpy(np.array(self.train_label[index])).type(torch.LongTensor)
        size = train_setting.size()
        train_setting = train_setting.view(1, size[2], size[3])
        label_setting = label_setting.view(-1)
        return train_setting, label_setting

    def __len__(self):
        return len(self.train)

# load train data and test data
train, train_label, _ = MERData(
                                cfg.datasets[0], cfg.datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=cfg.batch_Imagesize, maxlen=cfg.maxlen, maxImagesize=cfg.maxImagesize
                            )
len_train = len(train)

test, test_label, _ = MERData(
                                cfg.valid_datasets[0], cfg.valid_datasets[1], worddicts, batch_size=1,
                                batch_Imagesize=cfg.batch_Imagesize, maxlen=cfg.maxlen, maxImagesize=cfg.maxImagesize
                          )
len_test = len(test)

image_train = custom_dset(train, train_label)
image_test = custom_dset(test, test_label)

def collate_fn(batch):
    """ 引入掩码 双通道"""
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
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
    return img_padding_mask, label_padding

# def collate_fn(batch):
#     """ 不引入掩码 单通道 """
#     batch.sort(key=lambda x: len(x[1]), reverse=True) # 按图片大小排序
#     img, label = zip(*batch)

#     # 一个batch中最大的高宽
#     maxH = 0
#     maxW = 0
#     for j in range(len(img)):
#         size = img[j].size()
#         if size[1] > maxH:
#             maxH = size[1]
#         if size[2] > maxW:
#             maxW = size[2]

#     k = 0
#     for ii in img:
#         ii = ii.float()
#         img_size_h = ii.size()[1]
#         img_size_w = ii.size()[2]

#         # padding 图片
#         padding_h = maxH-img_size_h
#         padding_w = maxW-img_size_w
#         m = torch.nn.ConstantPad2d((0, padding_w, 0, padding_h), 255.)
#         img_sub_padding = m(ii)
#         img_sub_padding = img_sub_padding.unsqueeze(0)

#         if k==0:
#             img_padding = img_sub_padding
#         else:
#             img_padding = torch.cat((img_padding, img_sub_padding), dim=0)
#         k = k+1
        
#     max_len = len(label[0])+1  
#     k1 = 0
#     for ii1 in label:
#         ii1 = ii1.long()
#         ii1 = ii1.unsqueeze(0)
#         ii1_len = ii1.size()[1]
#         m = torch.nn.ZeroPad2d((0, max_len-ii1_len, 0, 0))
#         ii1_padding = m(ii1)
#         if k1 == 0:
#             label_padding = ii1_padding
#         else:
#             label_padding = torch.cat((label_padding, ii1_padding), dim=0)
#         k1 = k1+1

#     img_padding = img_padding/255.0
#     return img_padding, label_padding
    
train_loader = torch.utils.data.DataLoader(
    dataset = image_train,
    batch_size = cfg.batch_size,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = cfg.num_workers,
    )
    
test_loader = torch.utils.data.DataLoader(
    dataset = image_test,
    batch_size = cfg.batch_size_t,
    shuffle = True,
    collate_fn = collate_fn,
    num_workers = cfg.num_workers,
)

# load model
encoder = Encoder(img_channels=2)
decoder = Decoder(cfg.num_class)

# load pre-train
# encoder_dict = torch.load('checkpoints/encoder_3_19.pkl')
# encoder.load_state_dict(encoder_dict)
# decoder_dict = torch.load('checkpoints/attn_decoder_3_19.pkl')
# decoder.load_state_dict(decoder_dict)

encoder = encoder.cuda()
decoder = decoder.cuda()
# encoder = torch.nn.DataParallel(encoder, device_ids=cfg.gpu)
# decoder = torch.nn.DataParallel(decoder, device_ids=cfg.gpu)

# 定义损失函数,优化器和学习率策略
criterion = nn.CrossEntropyLoss().cuda()
encoder_optimizer = optim.SGD(encoder.parameters(), lr=cfg.lr, momentum=0.9)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=cfg.lr, momentum=0.9)
# encoder_optimizer = optim.Adadelta(encoder.parameters(), lr=cfg.lr, weight_decay=10e-4)
# decoder_optimizer = optim.Adadelta(decoder.parameters(), lr=cfg.lr, weight_decay=10e-4)
# scheduler_encoder = optim.lr_scheduler.MultiStepLR(encoder_optimizer, [20, 40], gamma=0.1)
# scheduler_decoder = optim.lr_scheduler.MultiStepLR(encoder_optimizer, [20, 40], gamma=0.1)

for epoch in range(cfg.num_epoch):
    running_loss=0
    whole_loss = 0

    encoder.train(mode=True)
    decoder.train(mode=True)

    # 开始训练 
    for step, (x, y) in enumerate(train_loader):
        if x.size()[0] < cfg.batch_size:  
            break
        x = x.cuda()
        y = y.cuda()
        # ----编码部分----
        low_feature, high_feature = encoder(x)
        
        # ----解码部分----
        # 初始化decoder输入, 隐藏层
        decoder_input = torch.LongTensor([111]*cfg.batch_size).view(-1, 1).cuda()       
        decoder_hidden = decoder.init_hidden(cfg.batch_size).cuda()
        # 重置coverage
        decoder.reset(cfg.batch_size, low_feature.size(), high_feature.size())

        target_length = y.size()[1]
        loss = 0

        # 是否使用tf用于RNN训练

        use_teacher_forcing = True if random.random() < cfg.teacher_forcing_ratio else False 
        flag_z = [0]*cfg.batch_size
        
        if use_teacher_forcing:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, low_feature, high_feature)

                y = y.unsqueeze(0)
                for i in range(cfg.batch_size):
                    if int(y[0][i][di]) == 0:
                        flag_z[i] = flag_z[i]+1
                        if flag_z[i] > 1:
                            continue
                        else:
                            loss += criterion(decoder_output[i].view(1, -1), y[:,i,di])
                    else:
                        loss += criterion(decoder_output[i].view(1, -1), y[:,i,di])

                if int(y[0][0][di]) == 0:
                    break
                decoder_input = y[:,:,di].transpose(1, 0)
                y = y.squeeze(0)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
        else:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            for di in range(target_length):
                decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, low_feature, high_feature)

                topv, topi = torch.max(decoder_output, 1)
                decoder_input = topi
                decoder_input = decoder_input.view(cfg.batch_size, 1)

                y = y.unsqueeze(0)
                for k in range(cfg.batch_size):
                    if int(y[0][k][di]) == 0:
                        flag_z[k] = flag_z[k]+1
                        if flag_z[k] > 1:
                            continue
                        else:
                            loss += criterion(decoder_output[k].view(1, -1), y[:,k,di])
                    else:
                        loss += criterion(decoder_output[k].view(1, -1), y[:,k,di])
                y = y.squeeze(0)
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            running_loss += loss.item()
    
        if step % 20 == 19:
            pre = ((step+1)/len_train)*100*cfg.batch_size
            whole_loss += running_loss
            running_loss = running_loss/(cfg.batch_size*20)
            print('epoch is %d, lr rate is %.5f, te is %.3f, batch_size is %d, loading for %.3f%%, running_loss is %f' %(epoch,cfg.lr,cfg.teacher_forcing_ratio, cfg.batch_size,pre,running_loss))
            running_loss = 0

    loss_all_out = whole_loss / len_train
    print("epoch is %d, the whole loss is %f" % (epoch, loss_all_out))

    # this is the prediction and compute wer loss
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    whole_loss_t = 0

    # ------- 验证 ----------
    encoder.eval()
    decoder.eval()
    print('Now, begin testing!!')

    for step_t, (x_t, y_t) in enumerate(test_loader):
        x_real_high = x_t.size()[2]
        x_real_width = x_t.size()[3]

        # 丢弃小于batch大小的数据
        if x_t.size()[0]<cfg.batch_size_t:
            break

        print('testing for %.3f%%'%(step_t*100*cfg.batch_size_t/len_test),end='\r')
        
        x_t = x_t.cuda()
        y_t = y_t.cuda()

        low_feature_t, high_feature_t = encoder(x_t)
        # 初始化输入
        decoder_input_t = torch.LongTensor([111]*cfg.batch_size_t).view(-1, 1).cuda()
        decoder_hidden_t = decoder.init_hidden(cfg.batch_size_t).cuda()     
        # 重置coverage  
        decoder.reset(cfg.batch_size_t, low_feature_t.size(), high_feature_t.size())

        prediction = torch.zeros(cfg.batch_size_t, cfg.maxlen)
        prediction_sub = []
        label_sub = []

        m = torch.nn.ZeroPad2d((0, cfg.maxlen-y_t.size()[1], 0, 0))
        y_t = m(y_t)
        for i in range(cfg.maxlen):
            decoder_output_t, decoder_hidden_t, _ = decoder(decoder_input_t, decoder_hidden_t, low_feature_t, high_feature_t)
            topv, topi = torch.max(decoder_output_t, 1)
            if torch.sum(topi) == 0:
                break
            decoder_input_t = topi
            decoder_input_t = decoder_input_t.view(cfg.batch_size_t, 1)

            # prediction
            prediction[:, i] = decoder_input_t.flatten()

        for i in range(cfg.batch_size_t):
            for j in range(cfg.maxlen):
                if int(prediction[i][j]) ==0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
            if len(prediction_sub)<cfg.maxlen:
                prediction_sub.append(0)

            for k in range(y_t.size()[1]):
                if int(y_t[i][k]) ==0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
            total_dist += dist
            total_label += llen
            total_line += 1
            # dist=0表示公式完全识别
            if dist == 0:
                total_line_rec = total_line_rec+ 1

            label_sub = []
            prediction_sub = []

    print('total_line_rec is', total_line_rec)
    wer = float(total_dist) / total_label
    sacc = float(total_line_rec) / total_line
    print('wer is %.5f' % (wer))
    print('sacc is %.5f ' % (sacc)) # ExpRate
    # 保存模型
    if (wer < best_wer):
        exprate = sacc
        best_wer = wer
        print('currect ExpRate:{}'.format(exprate))
        print("saving the model....")
        torch.save(encoder.state_dict(), 'checkpoints/encoder_.pkl')
        torch.save(decoder.state_dict(), 'checkpoints/attn_decoder_.pkl')
        print("done")
    else:
        print('the best is %f' % (exprate))
        print('the best wer is {}'.format(wer))
        print('the loss is bigger than before,so do not save the model')









