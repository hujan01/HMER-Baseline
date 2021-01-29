'''
Author: sigmoid
Description: 
Email: 595495856@qq.com
Date: 2020-10-21 21:20:51
LastEditTime: 2020-12-29 20:35:21
'''
import os
import sys

import pickle as pkl
import numpy

error_latex_file = 'errorlatex.txt'
with open(error_latex_file) as fr:
    error_list = [ x.strip() for x  in fr.readlines()]

fw = open('data/label/test_caption_2016.txt', 'w')
with open('data/test2016.txt') as f:
    lines = f.readlines()
    for process_num, line in enumerate(lines):
        parts = line.split()
        key = parts[0]
        if key not in error_list:
            fw.write(line)
            continue
        raw_cap = parts[1:]
        cap = []  
        idx = 0
        while idx<len(raw_cap) :
            if (idx!=len(raw_cap)-1) and (raw_cap[idx] in ['_', '^']) and (raw_cap[idx+1] != '{'):
                cap.append(raw_cap[idx])
                cap.append('{')
                cap.append(raw_cap[idx+1])
                cap.append('}')
                idx += 1
            else:
                cap.append(raw_cap[idx])
            idx += 1
        fw.write(key+'\t'+' '.join(cap)+'\n')
        cap = []
fw.close()