'''
@Author: your name
@Date: 2020-01-06 15:08:07
@LastEditTime: 2020-06-10 23:02:56
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /HMER/config.py
'''

class Config():  
    seed = 2020
    
    datasets = ['train.pkl', 'data/train_caption.txt']
    valid_datasets = ['valid.pkl', 'data/test_caption.txt']
    dictionaries = 'data/dictionary.txt'

    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000 
    maxImagesize = 100000

    maxlen = 70
    hidden_size = 256
    num_class = 112

    num_epoch = 70
    lr = 0.0001
    batch_size = 2
    batch_size_t = 2
    teacher_forcing_ratio = 0.8

    gpu = [0]
    num_workers = 4

cfg = Config()