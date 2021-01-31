'''
@Author: your name
@Date: 2020-01-06 15:08:07
<<<<<<< HEAD
LastEditTime: 2021-01-31 00:14:15
=======
LastEditTime: 2021-01-29 19:14:54
>>>>>>> 391ac2010bd4d47ccbed8a59065bfabc5166a8c3
LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /HMER/config.py
'''

class Config():  
    seed = 2020
    
    datasets = ['data/train_v1.pkl', 'data/label/train_caption_normal.txt']
    valid_datasets = ['data/valid_v1.pkl', 'data/label/test_caption_normal.txt']
    dictionaries = 'data/dictionary109.txt'

    batch_Imagesize = 500000
    valid_batch_Imagesize = 500000 
    maxImagesize = 100000

    maxlen = 70
    hidden_size = 256
    num_class = 112

    num_epoch = 80
    lr = 0.00005
    batch_size = 4
    batch_size_t = 4
    teacher_forcing_ratio = 0.8
 
    num_workers = 4

cfg = Config()