'''
Author: your name
Date: 2020-10-21 21:20:51
LastEditTime: 2020-10-21 21:28:29
LastEditors: your name
Description: In User Settings Edit
FilePath: /HMER/norm_dic.py
'''
fw = open("dictionnary_107.txt", "w")

with open("data/dictionary.txt") as f :
    lines = f.readlines()

    for idx, l in enumerate(lines, 1):
        x = l.split('\t')[0]
        s = x+'\t'+str(idx)+"\n"
        fw.write(s);

fw.close();