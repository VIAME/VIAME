# 
import numpy as np
import cv2

def create_training(flowList,selList,imgRoi=None,border=9,heatImg=None,heatThres=0.1,posRate=0.6,negRate=0.15):
    """ 
    flowList: roi list from flow segmentation label,(x,y,w,h)
    selList:  roi list from selective search  v,(x,y,w,h)
    validRoi: x,y,w,h
    return: positiveRoiList,negativeRoiList
    """
    pos,neg,posFlow,negFlow,ignores = [],[],[],[],[]

    # seperate flowList to posFlow,negFlow
    for vroi in flowList:
        label,roi = vroi
        if label == 0:
            negFlow.append(roi)
        else:
            posFlow.append(roi)

    print "negFlow=%d posFlow=%d"%(len(negFlow),len(posFlow))

    for roi in selList:
        if not roiValid(roi,imgRoi,border):
            continue

        rate = overlaps(roi,posFlow)
        if rate > posRate:
            pos.append(roi)
        elif rate < negRate:
            if not isOnHeat(roi,heatImg,heatThres):
                neg.append(roi)
            else:
                ignores.append(roi)
        else:
            ignores.append(roi)

    print "neg=%d pos=%d"%(len(neg),len(pos))

    return pos,neg,negFlow,ignores

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

def isOnHeat(roi,heatMap,threshold):
    if heatMap == None:
        return False
    x,y,w,h = roi
    dec = 2
    roiImg = heatMap[y/dec:(y+h)/dec,x/dec:(x+w)/dec] # heatMap is decimated
    r = np.mean(roiImg)/255.0
    print "Heat=%f"%(r)
    return  r > threshold
        

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

class MotionColor:
    def __init__(self):
        self.q = []

    def get(self,img,ch=0,temp=1):
        self.q.insert(0,img) # add to first
        qSize = 2*temp+1
        if len(self.q) < qSize:
            return img
        while len(self.q) > qSize:
            self.q.pop() # remove last
        b = self.q[0][:,:,ch]
        g = self.q[temp][:,:,ch]
        r = self.q[2*temp][:,:,ch]
        return cv2.merge((b,g,r))




