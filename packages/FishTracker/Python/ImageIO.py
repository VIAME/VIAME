import numpy
import cv2
import os

def mkdir_p(path):
    """ mkdir -p """
    if not os.path.exists(path):
        dir = os.path.dirname(path)
        mkdir_p(dir)
        os.makedirs(path)

def crop(img,roi,border):
    height, width, channels = img.shape
    if roi == None:
        return img
    else:
        (x,y,w,h) = roi
        x = x - border
        y = y - border
        w = w + 2*border
        h = h + 2*border
        if x<0:
            x=0
        if y<0:
            y=0
    	if x + w > width:
	    w = width - x
    	if y + h > height:
	    h = height - y
        return img[y:y+h,x:x+w]

def saveImg(path,img,roi=None,border=0):
    """ create all directories in path and save image """
    dir = os.path.dirname(path)
    mkdir_p(dir)
    cv2.imwrite(path,crop(img,roi,border))

