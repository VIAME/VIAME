#!/usr/bin/env python
import numpy as np
import cv2
import sys
import os
from sys import argv
from os import listdir
from os.path import isfile, join
import glob
from selective_search import *

def list_dirs(path):
    # returns a list of subfolders
    dirs = []
    for name in os.listdir(path):
        if os.path.isdir(os.path.join(path, name)):
            dirs.append(name)
    return dirs 
    
def list_files(path):
    # returns a list of names (with extension, without full path) of all files 
    # in folder path
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files 

def roiValid(roi,imgRoi,border):
    if imgRoi == None:
        return True
    (x,y,w,h) = roi
    (X,Y,W,H) = imgRoi
    if x < X+border:
        return False
    if x+w > X+W-border:
        return False
    if y < Y+border :
        return False
    if y+h > Y+H-border:
        return False
    return True
    
def removeDuplicates(rois,rate=0.95):
    output = []
    for i in range(len(rois)):
        if overlaps(rois[i],rois[i+1:]) < rate:
            output.append(rois[i])
    return output

def removeSize(rois,width,height,max,min):
    output = []
    for roi in rois:
        (x,y,w,h) = roi
        r = float(w)/float(width)
        if  r > max or r < min:
            continue;
        r = float(h)/float(height)
        if  r > max or r < min:
            continue;
        output.append(roi)
    return output

def removeInvalid(rois,width,height,border=9):
    imgRoi = (0,0,width,height)
    output = []
    for roi in rois:
        if roiValid(roi,imgRoi,border):
            output.append(roi)
    return output

def overlap(roi1,roi2):
    """ calculate overlap rate 
    roi1,roi2: (x,y,w,h)
    """
    (x1,y1,w1,h1) = roi1
    (x2,y2,w2,h2) = roi2
    area1,area2 = w1*h1, w2*h2
    xx1,yy1 = max(x1,x2),max(y1,y2)
    xx2,yy2 = min(x1+w1,x2+w2),min(y1+h1,y2+h2)
    w,h = max(0,xx2-xx1),max(0,yy2-yy1)
    inter = (float)(w*h)
    if area1+area2-inter == 0:
        return 0.0
    else :
        return inter/(area1+area2-inter)

def overlaps(roi,rois):
    """ calculate max rate of overlap between roi and rois
    vroi: (x,y,w,h)
    vrois:(x,y,w,h)
    """
    max = 0
    label = 1
    for r in rois:
        cur = overlap(roi,r)
        if cur >= max:
            max = cur
    return max
    
def nms_area(rois, overlap):
    box_list = []
    if len(rois)==0: 
	return []
    for i in range(len(rois)):
        roi = rois[i]
        area_roi = roi[2] * roi[3]
        box = [roi[0], roi[1], roi[0]+roi[2]-1, roi[1]+roi[3]-1, area_roi]
        box_list.append(box)        
    boxes = np.array(box_list)   
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    a  = boxes[:,4]
    
    area = (x2-x1+1) * (y2-y1+1)

    I = np.argsort(a)
 
    pick = np.zeros(a.size, dtype=np.int)
    counter = 0

    while I.size > 0:
        last = I.size-1
        i = I[-1]
        pick[counter] = i
        counter += 1
       
        xx1 = np.maximum(x1[i], x1[I[:-1]])
        yy1 = np.maximum(y1[i], y1[I[:-1]])
        xx2 = np.minimum(x2[i], x2[I[:-1]])
        yy2 = np.minimum(y2[i], y2[I[:-1]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)


        o = w*h / area[I[:-1]]
        np.nonzero(o > overlap)[0]
        
        I = np.delete(I, np.concatenate([[last], np.nonzero(o > overlap)[0]]))
        
    pick = pick[:counter]
    top = boxes[pick,:]
    return top
    
def create_samples (img):
 
    width  = img.shape[1]
    height = img.shape[0]
    regions = selective_search(img,
                               color_spaces = ['rgb'], 
                               ks = [200],
                               feature_masks=[features.SimilarityMask(size=True, 
                                                                      color=True, 
                                                                      texture=True, 
                                                                      fill=True)])                                                                  
    selRois = [(x1,y1,x2-x1+1,y2-y1+1) for v,(y1,x1,y2,x2) in regions]
    selRois = removeDuplicates(selRois,0.75)
    selRois = removeInvalid(selRois,width,height)
    selRois = removeSize(selRois,width,height,0.5,0.015)
    selRois = nms_area(selRois, 0.4)
    
    return selRois 

if __name__ == "__main__":
    inDir = os.path.normpath('/media/david/DZhang_2012/NOVA_videos/MBARI_VIDEO/MBARI_V3165-01')
    outDir = os.path.normpath('/media/david/DZhang_2012/NOVA_videos/MBARI_VIDEO/MBARI_V3165-01/Z_Samples')
    onlyDirs = list_dirs(inDir)
    cv2.namedWindow("Source",cv2.WINDOW_AUTOSIZE);
    
    for f in onlyDirs:
        #if f < 'sea_star' or f > 'sea_urchin' :
	if f != 'tests':
            continue
        subdir = os.path.join(inDir, f)
        onlyfiles = sorted(list_files(subdir))
        subdir_out = os.path.join(outDir, f)
        if not os.path.exists(subdir_out):
            os.makedirs(subdir_out)
               
        for fimg in onlyfiles:
            jj = 0
            imgname = os.path.join(subdir, fimg)
            img = cv2.imread(imgname)
            height, width = img.shape[:2]
            img_lin = img
 
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
            cv2.normalize(img, img_lin, 0, 255,  cv2.NORM_MINMAX)
            #half of the original size
            img_half = cv2.resize(img_lin, (width/2, height/2), interpolation=cv2.INTER_CUBIC)
            rois = create_samples(img_half)
            print fimg
            print rois
            
            for (x0, y0, x1, y1, area) in rois:
                x0 = np.maximum(0, 2*x0-30)
                y0 = np.maximum(0, 2*y0-30)
                x1 = np.minimum(img.shape[1], 2*x1+30)
                y1 = np.minimum(img.shape[0], 2*y1+30)               
                roi_img = img_lin[y0:y1,x0:x1]
                roi_fl_1 = os.path.splitext(fimg)
                roi_fl_2 = '%s_%06d%s' % (roi_fl_1[0], jj, '.jpeg')
                roi_file = os.path.join(subdir_out, roi_fl_2)
                cv2.imwrite(roi_file, roi_img)
                jj+=1
                
            for (x0, y0, x1, y1, area) in rois:
                cv2.rectangle(img_half,(x0,y0),(x1,y1),(0,255,0),3)
                                                             
            cv2.imshow("Source",img_half)
            cv2.waitKey(200)    
    cv2.destroyAllWindows()
