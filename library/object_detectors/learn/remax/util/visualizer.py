# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

# -*- coding: utf-8 -*-
'''
@File    :   visualizer.py
@Time    :   2022/04/05 11:39:33
@Author  :   Shilong Liu 
@Contact :   liusl20@mail.tsinghua.edu.cn; slongliu86@gmail.com
Modified from COCO evaluator
'''

import os, sys
from textwrap import wrap
import torch
import numpy as np
import cv2
import datetime
import pdb
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from pycocotools import mask as maskUtils
from matplotlib import transforms
import utils
def renorm(img: torch.FloatTensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
        -> torch.FloatTensor:
    # img: tensor(3,H,W) or tensor(B,3,H,W)
    # return: same as img
    assert img.dim() == 3 or img.dim() == 4, "img.dim() should be 3 or 4 but %d" % img.dim() 
    if img.dim() == 3:
        assert img.size(0) == 3, 'img.size(0) shoule be 3 but "%d". (%s)' % (img.size(0), str(img.size()))
        img_perm = img.permute(1,2,0)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(2,0,1)
    else: # img.dim() == 4
        assert img.size(1) == 3, 'img.size(1) shoule be 3 but "%d". (%s)' % (img.size(1), str(img.size()))
        img_perm = img.permute(0,2,3,1)
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        img_res = img_perm * std + mean
        return img_res.permute(0,3,1,2)

class ColorMap():
    def __init__(self, basergb=[255,255,0]):
        self.basergb = np.array(basergb)
    def __call__(self, attnmap):
        # attnmap: h, w. np.uint8.
        # return: h, w, 4. np.uint8.
        assert attnmap.dtype == np.uint8
        h, w = attnmap.shape
        res = self.basergb.copy()
        res = res[None][None].repeat(h, 0).repeat(w, 1) # h, w, 3
        attn1 = attnmap.copy()[..., None] # h, w, 1
        res = np.concatenate((res, attn1), axis=-1).astype(np.uint8)
        return res


class COCOVisualizer():
    def __init__(self) -> None:
        self.colors = utils.MOUSE_10X_COLORS
        pass

    def visualize1(self, img, tgt, tpred, results, caption=None, dpi=120, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        
        self.addtgt(tgt, tpred, results)

        if savedir is not None:
            if caption is None:
                savename = '{}/img{}.png'.format(savedir, int(tgt['image_id']))
            else:
                savename = '{}/img{}-{}.png'.format(savedir, int(tgt['image_id']), caption)
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename)
        plt.close()

    def visualize(self, img, tgt, tpred, caption=None, dpi=120, savedir=None, show_in_console=True):
        """
        img: tensor(3, H, W)
        tgt: make sure they are all on cpu.
            must have items: 'image_id', 'boxes', 'size'
        """
        plt.figure(dpi=dpi)
        plt.rcParams['font.size'] = '5'
        ax = plt.gca()
        img = renorm(img).permute(1, 2, 0)
        ax.imshow(img)
        
        self.addtgt(tgt, tpred)

        if savedir is not None:
            if caption is None:
                savename = '{}/img{}.png'.format(savedir, int(tgt['image_id']))
            else:
                savename = '{}/img{}-{}.png'.format(savedir, int(tgt['image_id']), caption)
            print("savename: {}".format(savename))
            os.makedirs(os.path.dirname(savename), exist_ok=True)
            plt.savefig(savename)
        plt.close()

    def addtgt(self, tgt, tpred):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox_gt = tgt['boxes'].shape[0]
        numbox_pred = tpred['boxes'].shape[0]

        color = []
        polygons_gt, polygons_pred_tp, polygons_pred_fp = [], [], []
        polygons_pred_evm, polygons_pred_softmax = [], []
        boxes_gt, boxes_pred = [], []
        for k in range(numbox_gt):
            box = tgt['boxes'][k].cpu()
            label = tgt['gt_label'][k]               
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes_gt.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons_gt.append(Polygon(np_poly))
            c = self.colors[label] #(np.random.random((1, 3))*0.6+0.4).tolist()[0]
            color.append(c)

        for k in range(numbox_pred):
            box = tpred['boxes'][k].cpu()  
            label = tpred['pred_label'][k]           
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes_pred.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            if k in tpred['tp_idx']:
                polygons_pred_tp.append(Polygon(np_poly))
            else:
                polygons_pred_fp.append(Polygon(np_poly))
                            
        p = PatchCollection(polygons_gt, facecolor='orange', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_gt, facecolor='none', edgecolors='orange', linewidths=1.5)
        ax.add_collection(p)

        p = PatchCollection(polygons_pred_fp, facecolor='blue', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_pred_fp, facecolor='none', edgecolors='blue', linewidths=1)
        ax.add_collection(p)

        p = PatchCollection(polygons_pred_tp, facecolor='green', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_pred_tp, facecolor='none', edgecolors='green', linewidths=1)
        ax.add_collection(p)

        ax.set_title("GT (orange) TP (green) FP (blue)")
        cnt = 0
        if 'box_label' in tgt:
            for bl, label in zip(tgt['box_label'], tgt['gt_label']):
                _string = str(bl) 
                bbox_x, bbox_y, bbox_w, bbox_h = boxes_gt[cnt]
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[cnt], 'alpha': 0.6, 'pad': 1})
                cnt += 1


    def addtgt1(self, tgt, tpred, results):
        """
        - tgt: dict. args:
            - boxes: num_boxes, 4. xywh, [0,1].
            - box_label: num_boxes.
        """
        assert 'boxes' in tgt
        ax = plt.gca()
        H, W = tgt['size'].tolist() 
        numbox_gt = tgt['boxes'].shape[0]
        numbox_pred = tpred['boxes'].shape[0]

        color = []
        polygons_gt, polygons_pred_tp, polygons_pred_fp = [], [], []
        polygons_pred_evm, polygons_pred_softmax = [], []
        boxes_gt, boxes_pred = [], []
        for k in range(numbox_gt):
            box = tgt['boxes'][k].cpu()
            label = tgt['gt_label'][k]
            #if not label in [14,15,16,17,18,19,20,21,42]:
            #    continue
                
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes_gt.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            polygons_gt.append(Polygon(np_poly))
            c = self.colors[label] #(np.random.random((1, 3))*0.6+0.4).tolist()[0]
            color.append(c)

        for k in range(numbox_pred):
            box = tpred['boxes'][k].cpu()  
            label = tpred['pred_label'][k]
            evm_sc = results['evm_score'][k]
            softmax_sc = results['softmax_score'][k]
            if evm_sc > 0.75 and softmax_sc < 0.68: 
                label = 42
            #if not label in [14,15,16,17,18,19,20,21]:
            #    continue             
            unnormbbox = box * torch.Tensor([W, H, W, H])
            unnormbbox[:2] -= unnormbbox[2:] / 2
            [bbox_x, bbox_y, bbox_w, bbox_h] = unnormbbox.tolist()
            boxes_pred.append([bbox_x, bbox_y, bbox_w, bbox_h])
            poly = [[bbox_x, bbox_y], [bbox_x, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y+bbox_h], [bbox_x+bbox_w, bbox_y]]
            np_poly = np.array(poly).reshape((4,2))
            if k in tpred['tp_idx']:
                if evm_sc > 0.75:
                    print("evm ", evm_sc)
                if softmax_sc > 0.68:
                    print("softmax ", softmax_sc)
                polygons_pred_tp.append(Polygon(np_poly))
            else:
                polygons_pred_fp.append(Polygon(np_poly))
                            
        p = PatchCollection(polygons_gt, facecolor='orange', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_gt, facecolor='none', edgecolors='orange', linewidths=1.5)
        ax.add_collection(p)

        p = PatchCollection(polygons_pred_fp, facecolor='blue', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_pred_fp, facecolor='none', edgecolors='blue', linewidths=1)
        ax.add_collection(p)

        p = PatchCollection(polygons_pred_tp, facecolor='green', linewidths=0, alpha=0.1)
        ax.add_collection(p)
        p = PatchCollection(polygons_pred_tp, facecolor='none', edgecolors='green', linewidths=1)
        ax.add_collection(p)

        ax.set_title("GT (orange) TP (green) FP (blue)")
        cnt = 0
        if 'box_label' in tgt:
            for bl, label in zip(tgt['box_label'], tgt['gt_label']):
                _string = str(bl)
                #if not label in [14,15,16,17,18,19,20,21,42]:
                #    continue 
                bbox_x, bbox_y, bbox_w, bbox_h = boxes_gt[cnt]
                ax.text(bbox_x, bbox_y, _string, color='black', bbox={'facecolor': color[cnt], 'alpha': 0.6, 'pad': 1})
                cnt += 1

