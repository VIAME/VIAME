from __future__ import division
import itertools
import os
import random 
import numpy as np
import sys
import json

_, label_file = sys.argv
reader = open(os.path.join(os.environ['FS_ROOT'], label_file), 'rt')
label = json.load(reader)
reader.close()

path = label['data_path']
pos_dics = label['pos_dics']
neg_dics = label['neg_dics']
num_types = len(pos_dics)
train_file = os.path.join(os.environ['FS_ROOT'], os.path.dirname(label_file), 'train.txt')
test_file = os.path.join(os.environ['FS_ROOT'], os.path.dirname(label_file), 'test.txt')

pos_list = {}
for element in pos_dics :
    label = pos_dics[element]     
    if str(element) == 'negative':
        continue
    pname = os.path.join(path, element)
    imglist = os.listdir(pname) 
    temp = [os.path.join(pname, img) for img in imglist] 
    random.shuffle(temp)
    pos_list[label] = temp

neg_list = {}
for element in neg_dics :
    label = neg_dics[element]     
    pname = os.path.join(path, element)
    imglist = os.listdir(pname) 
    temp = [os.path.join(pname, img) for img in imglist] 
    neg_list[label] = temp
    random.shuffle(temp)

max_num = 0
for name, idx in pos_dics.iteritems() :
    print 'label %d type %s size=%d' % (idx, name, len(pos_list[idx]))
    max_num = max(max_num, len(pos_list[idx]))
   
f = open(train_file, "w")    
for i in range(int(max_num*3/4)) :
    for element in pos_dics :
        label = pos_dics[element] 
        imglist = pos_list[label]
        k = (i%len(imglist))
        f.write("%s %d\n" % (imglist[k], label))
    for element in neg_dics :    
        label = neg_dics[element] 
        imglist = neg_list[label]
        k = (i%len(imglist))
        f.write("%s %d\n" % (imglist[k], label))

f.close()


f = open(test_file, "w")
for i in range(int(max_num*3/4), max_num) :
    for element in pos_dics :
        label = pos_dics[element] 
        imglist = pos_list[label]
        k = (i%len(imglist))
        f.write("%s %d\n" % (imglist[k], label))
    for element in neg_dics :    
        label = neg_dics[element] 
        imglist = neg_list[label]
        k = (i%len(imglist))
        f.write("%s %d\n" % (imglist[k], label))

f.close()

