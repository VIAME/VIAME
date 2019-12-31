import os
import numpy as np
import pickle
from collections import OrderedDict

import xml.etree.ElementTree
import xmltodict
import numpy as np

import  matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import time

output_path = './imagenet_refine.pkl'



seq_home = '/home/ilchae/dataset/ILSVRC/'
train_list = [p for p in os.listdir(seq_home + 'Data/VID/train')]
seq_list = []
for num, cur_dir in enumerate(train_list):
    seq_list += [cur_dir + '/' + p for p in os.listdir(seq_home + 'Data/VID/train/' + cur_dir)]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

data = {}
completeNum = 0
for i,seqname in enumerate(seq_list):
    print(seqname)
    seq_path = seq_home + 'Data/VID/train/' + seqname
    gt_path = seq_home +'Annotations/VID/train/' + seqname
    img_list = sorted([p for p in os.listdir(seq_path) if os.path.splitext(p)[1] == '.JPEG'])

    # gt = np.zeros((len(img_list),4))
    enable_gt = []
    enable_img_list = []
    gt_list = sorted([gt_path + '/' + p for p in os.listdir(gt_path) if os.path.splitext(p)[1] == '.xml'])
    save_enable = True
    for gidx in range(0,len(img_list)):
        with open(gt_list[gidx]) as fd:
            doc = xmltodict.parse(fd.read())
        try:
            try:
                object =doc['annotation']['object'][0]
            except:
                object = doc['annotation']['object']
        except:
            ## no object, occlusion and hidden etc.
            continue

        if (int(object['trackid']) is not 0):
            continue

        xmin = float(object['bndbox']['xmin'])
        xmax = float(object['bndbox']['xmax'])
        ymin = float(object['bndbox']['ymin'])
        ymax = float(object['bndbox']['ymax'])

        ## discard too big object
        if ((float(doc['annotation']['size']['width'])/2.) < (xmax-xmin) ) and ((float(doc['annotation']['size']['height'])/2.) < (ymax-ymin) ):
            continue

        # gt[gidx,0] = xmin
        # gt[gidx,1] = ymin
        # gt[gidx,2] = xmax - xmin
        # gt[gidx,3] = ymax - ymin

        cur_gt = np.zeros((4))
        cur_gt[0] = xmin
        cur_gt[1] = ymin
        cur_gt[2] = xmax - xmin
        cur_gt[3] = ymax - ymin
        enable_gt.append(cur_gt)

        enable_img_list.append(img_list[gidx])

    if len(enable_img_list) == 0:
        save_enable = False
    if save_enable:
        assert len(enable_img_list) == len(enable_gt), "Lengths do not match!!"
        data[seqname] = {'images':enable_img_list, 'gt':np.asarray(enable_gt)}
        completeNum += 1
        print 'Complete!'

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)

print 'complete {} videos'.format(completeNum)
