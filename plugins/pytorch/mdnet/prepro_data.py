import os
import numpy as np
import pickle
from collections import OrderedDict



# seq_home = '../dataset/'
seqlist_path = '../vot-otb.txt'
output_path = 'data/vot-otb.pkl'
set_type = 'VOT'
seq_home = '/home/ilchae/dataset/tracking/'+set_type +'/'

if set_type=='OTB':
    seqlist_path = '../otb-vot15.txt'
    output_path = '../otb-vot15.pkl'

if set_type == 'VOT':
    seqlist_path = '../vot-otb.txt'
    output_path = '../vot-otb.pkl'

with open(seqlist_path,'r') as fp:
    seq_list = fp.read().splitlines()

data = {}
for i,seqname in enumerate(seq_list):
    print(seqname)
    if set_type=='OTB':
        seq_path = seq_home+seqname
        img_list = sorted([p for p in os.listdir(seq_path+'/img') if os.path.splitext(p)[1] == '.jpg'])

        if (seqname == 'Jogging') or (seqname == 'Skating2'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect.1.txt')
        elif seqname == 'Human4' :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.2.txt', delimiter=',')
        elif (seqname == 'BlurBody') or (seqname == 'BlurCar1') or (seqname == 'BlurCar2') or (seqname == 'BlurCar3') \
                or (seqname == 'BlurCar4') or (seqname == 'BlurFace') or (seqname == 'BlurOwl') or (seqname == 'Board') \
                or (seqname == 'Box') or (seqname == 'Car4') or (seqname == 'CarScale') or (seqname == 'ClifBar') \
                or (seqname == 'Couple') or (seqname == 'Crossing') or (seqname == 'Dog') or (seqname == 'FaceOcc1') \
                or (seqname == 'Girl') or (seqname == 'Rubik') or (seqname == 'Singer1') or (seqname == 'Subway') \
                or (seqname == 'Surfer') or (seqname == 'Sylvester') or (seqname == 'Toy') or (seqname == 'Twinnings') \
                or (seqname == 'Vase') or (seqname == 'Walking') or (seqname == 'Walking2') or (seqname == 'Woman') :
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt')
        elif (seqname == 'Diving'):
            gt = np.loadtxt(seq_path + '/groundtruth_rect_ilchae.txt', delimiter=',')
        else:
            gt = np.loadtxt(seq_path + '/groundtruth_rect.txt', delimiter=',')

        if (seqname == 'David') or (seqname == 'Football1') or (seqname == 'Freeman3') or (seqname == 'Freeman4'):
            continue

    if set_type =='VOT':
        img_list = sorted([p for p in os.listdir(seq_home + seqname) if os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_home + seqname + '/groundtruth.txt', delimiter=',')

    if set_type == 'IMAGENET':
        img_list = []
        gt = []

    assert len(img_list) == len(gt), "Lengths do not match!!"

    if gt.shape[1]==8:
        x_min = np.min(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_min = np.min(gt[:,[1,3,5,7]],axis=1)[:,None]
        x_max = np.max(gt[:,[0,2,4,6]],axis=1)[:,None]
        y_max = np.max(gt[:,[1,3,5,7]],axis=1)[:,None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seqname] = {'images':img_list, 'gt':gt}

with open(output_path, 'wb') as fp:
    pickle.dump(data, fp, -1)
