#!/usr/bin/python
import os,sys,json,cv2,random 
import numpy as np
from selective_search import *
from FlowSeg import *
import ImageIO

class Annotate:
    def __init__(self,config,wrkDir = None):
        if wrkDir == None:
            wrkDir  = config["workDir"]
        self.wrkDir = os.path.abspath(wrkDir)

        self.config  = config["annotate"]
        self.border  = self.config["border"]
        self.fs = None
        self.imgRoi = [0,0,640,480]

    def selectiveSearch(self,img):
        """ return list of rois (x,y,w,h) """
        config = self.config["SelectiveSearch"]
        self.imgRoi = [0,0,img.shape[1],img.shape[0]]

        regions = selective_search(img,color_spaces = ['lab'], 
                                   ks = [config["ks"]],
                                   feature_masks=[features.SimilarityMask(
                                       size=config["size"], 
                                       color=config["color"], 
                                       texture=config["texture"], 
                                       fill=config["fill"])])

        return [(x1,y1,x2-x1+1,y2-y1+1) for v,(y1,x1,y2,x2) in regions]

    def flowSegment(self,img):
        self.imgRoi = [0,0,img.shape[1],img.shape[0]]
        if self.fs == None:
            config = self.config["FlowSegment"]
            self.fs = FlowSeg()
            self.fs.init(img.shape[1],img.shape[0], # cols,rows
                         config["finlev"],
                         config["toplev"],
                         config["laplacian"],
                         config["lev_iterations"],
                         config["flow_quantize"])

        (negRois,posRois) = self.fs.run(img)
        return (negRois,posRois)

    def merge(self,selRois,flowPosRois,maskImg=None):
        """
        check overlap of selective search and flowRosRois
        > posRate(0.6) : pos
        < negRate(0.15): neg
        in between:      ignore
        """
        config = self.config["merge"]
        pos = []
        neg = []
        ignores = []
        for roi in selRois:
            if not self.roiValid(roi,self.imgRoi,self.border):
                continue

            rate = self.overlaps(roi,flowPosRois)
            if rate > config["posRate"]:
                pos.append(roi)
            elif rate < config["negRate"]:
                if not self.isOnHeat(roi,maskImg,config["maskThres"]):
                    neg.append(roi)
                else:
                    ignores.append(roi)
            else:
                ignores.append(roi)
                
        print "neg=%d pos=%d ignor=%d"%(len(neg),len(pos),len(ignores))
        return pos,neg,ignores

    def crop(self,video,fnum,img,roisDict):
        """
        {'<label>': (filename,fnum,roi)),...}
        """
        bname = os.path.splitext(os.path.basename(video))[0]
        imgIndex = {}
        for key,rois in roisDict.iteritems():
            for i,roi in enumerate(rois):
                path = self.wrkDir + '/TrainImgs/' \
                       + key + '/'
                fname = "%s_%04d_%04d.jpg"%(bname,fnum,i)
                ImageIO.saveImg(path+fname,img,roi,self.border)
                imgIndex[key] = (fname,fnum,roi)
            self.indexUpdate(imgIndex)

    def labeling(self,filename,fnum,img,maskImg=None):
        ssRois = self.selectiveSearch(img)
        fsNegRois,fsPosRois= self.flowSegment(img)

        pos,neg,ignores = self.merge(ssRois,fsPosRois,maskImg)
        self.crop(filename,fnum,img,{'pos':pos,'neg':neg,'flowNeg':fsNegRois})
        return pos,neg,fsNegRois

    def indexUpdate(self,index):
        """ update image index file """
        indexFile = self.wrkDir + '/TrainImgs/ImgIdx.json'
        if os.path.isfile(indexFile):
            with open(indexFile) as fd:
                allIdx = json.load(fd)
        else:
            allIdx = {}

        for key,value in index.iteritems():
            if key in allIdx:
                allIdx[key].append(value)
            else:
                allIdx[key] = [value]

        with open(indexFile,'w') as fd:
            json.dump(allIdx,fd)

    def imageSelection(self):
        """ create training list """
        config = self.config["imageSelection"]

        # load index file
        indexFile = self.wrkDir + '/TrainImgs/ImgIdx.json'
        with open(indexFile) as fd:
            allIdx = json.load(fd)

        # shuffle all lists
        minLen = None
        for key,list in allIdx.iteritems():
            random.shuffle(allIdx[key])
            curLen = len(list)
            if minLen == None:
                minLen = curLen
            elif minLen > curLen:
                minLen = curLen
                
        minLen *= config["testRatio"]

        # create train file
        self.saveTrainList(self.wrkDir+'/TrainImgs/train.txt',allIdx,minLen)
        # create test file
        self.saveTrainList(self.wrkDir+'/TrainImgs/test.txt',allIdx,0)

    def saveTrainList(self,filename,allIdx,minLeft):
        config = self.config["imageSelection"]
        f = open(filename,"w")
        while True:
            isDone = False
            for key,list in allIdx.iteritems():
                if len(list) <= minLeft and config[key+'Rate'] != 0:
                    isDone = True
                    break
            if isDone:
                break

            roll = random.random()
            selKey = None
            for key,list in allIdx.iteritems():
                rate = config[key+'Rate']
                if roll < rate:
                    selKey = key
                    break
                else:
                    roll -= rate
            if selKey != None:
                filename,fnum,roi = allIdx[selKey][0]
                filepath = self.wrkDir+'/TrainImgs/' + key + '/' + filename
                f.write("%s %d\n" % (filepath,config[selKey]))
                allIdx[selKey].pop(0)
        f.close()


    def roiValid(self,roi,imgRoi,border):
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

    def isOnHeat(self,roi,heatMap,threshold):
        if heatMap == None:
            return False
        x,y,w,h = roi
        dec = 2
        roiImg = heatMap[y/dec:(y+h)/dec,x/dec:(x+w)/dec] # heatMap is decimated
        r = np.mean(roiImg)/255.0
        #print "Heat=%f"%(r)
        return  r > threshold
        
    def removeDuplicates(self,rois,rate=0.95):
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
                continue
            r = float(h)/float(height)
            if  r > max or r < min:
                continue
            output.append(roi)
        return output

    def removeInvalid(self,rois,width,height,border=9):
        imgRoi = (0,0,width,height)
        output = []
        for roi in rois:
            if self.roiValid(roi,imgRoi,border):
                output.append(roi)
        return output

    def overlap(self,roi1,roi2):
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
    
    def overlaps(self,roi,rois):
        """ calculate max rate of overlap between roi and rois
        vroi: (x,y,w,h)
        vrois:(x,y,w,h)
        """
        max = 0
        label = 1
        for r in rois:
            cur = self.overlap(roi,r)
            if cur >= max:
                max = cur
        return max

    def getHeatImg(self,cap,thresh=128):
        if not self.config["HeatMap"]["enable"]:
            return None
        fnums = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if fnums > self.config["HeatMap"]["length"]:
            fnums = self.config["HeatMap"]["length"]
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        heatImg = None
        for f in xrange(fnums):
            ret,img = cap.read()
            if heatImg == None:
                heatImg = img
                heatImg = heatImg.astype(np.uint32)
            else:
                heatImg += img
        heatImg /= fnums
        heatImg = heatImg.astype(np.uint8)
        bImg = (255*(heatImg>thresh)).astype(np.uint8)
        return bImg

#######################################################
# module test                
def main(argv):
    if len(argv) <2:
        print '%s <videofile>'%argv[0]
    video = argv[1] # video filename
    
    curDir = os.path.dirname(__file__)
    configFile = curDir+'/../bin/config/sysConfig.json' 
    with open(configFile) as fd:
        config = json.load(fd)

    anno = Annotate(config)

    cap = cv2.VideoCapture(video)
    if cap.isOpened() == False:
        print "Cannot open input video!"
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fnums = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    maskImg = anno.getHeatImg(cap)

    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for f in xrange(50): #xrange(int(fnums)):
        ret,img = cap.read()

        pos,neg,flowNeg = anno.labeling(video,f,img,maskImg)

        for (x,y,w,h) in pos:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        for (x,y,w,h) in neg:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
        for (x,y,w,h) in flowNeg:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)

        print 'total=',fnums,'fnum=',f
        if maskImg != None:
            cv2.imshow("heat",maskImg)
        cv2.imshow("Source",img)
        cv2.waitKey(10)

    # may need to open multiple video files and then call imageSelection
    anno.imageSelection();


if __name__ == "__main__":
    main(sys.argv)
