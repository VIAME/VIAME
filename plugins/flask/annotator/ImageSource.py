#! /usr/bin/env python
import numpy as np
import json,cv2,os,time,glob,copy,re
from color import *

def cv2ver():
    major = cv2.__version__.split('.')[0]
    return int(major)

class ImageSource:
    EmptyData = {
        "frames": [],
        "input_file": ""
    }

    EmptyFrame = {
        "frame_id": 0, 
        "frame_rois": []
    }

    def __init__(self, videoname,jsonname):
        self.videoInputEnable = False
        self.roisInputEnable = False
        self.isAvi = False
        self.openVideo(videoname)
        self.openRoiJson(jsonname)

    def openVideo(self,videoname):
        self.videofile = videoname
        filename,ext = os.path.splitext(self.videofile)
        self.jsonfile = filename + '.json'
        ext = ext.lower()
        if ext == '.avi':
            self.openVideoAvi(videoname)
        elif ext == '.bmp' or ext == '.jpg' or ext == '.png':
            self.openVideoImg(videoname)
        
    def openVideoAvi(self,videoname):
        try:
            self.cap = cv2.VideoCapture(videoname)
            if cv2ver() == 3:
                self.fcnt = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            else:
                self.fcnt = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
            self.isAvi = True
            self.videoInputEnable = True
            self.createEmptyData(self.fcnt)
        except:
            self.videoInputEnable = False

    def file2list(self,filename):
        dirname = os.path.dirname(filename)
        basename = os.path.basename(filename)
        # replace digits with ? in basename
        chars = list(basename)
        newChars = []
        for c in chars:
            if c.isdigit():
                newChars.append('?')
            else:
                newChars.append(c)
        basename = ''.join(newChars)
        return sorted(glob.glob(dirname+'/'+basename))
        
            
    def openVideoImg(self,imgname):
        self.videoList = self.file2list(imgname)
        self.fcnt = len(self.videoList)
        self.videoInputEnable = True
        self.isAvi = False
        self.createEmptyData(self.fcnt)

    def openImgDir(self,dirname):
        filelist = sorted(glob.glob(dirname+'/*'))
        if len(filelist) > 0:
            self.openVideoImg(filelist[0])
        
        
    def openRoiJson(self,jsonname):
        self.jsonfile = jsonname
        try:
            with open(jsonname) as fd:
                self.data = json.load(fd)
            self.fcntRois = len(self.data['frames'])
            self.roisInputEnable = True
        except:
            self.roisInputEnable = False
        if self.roisInputEnable:
            self.labels = self.getLabelFromData(self.data)
        else:
            self.labels = [('Unsorted',-1),
                           ('Negative',0)]

    def openRoiTxt(self,txtname):
        txtlist = self.file2list(txtname)
        if self.fcnt != len(txtlist):
            print('Error: mismatch')
            return
        self.createEmptyData(self.fcnt)
        frameid = 0
        for afile in txtlist:
            with open(afile) as fd:
                lines = fd.readlines()
            roiid = 0
            for line in lines:
                words = line.split()
                try:
                    (x,y,w,h) = (int(words[0]),int(words[1]),int(words[2]),int(words[3]))
                except:
                    continue
                self.data['frames'][frameid]['frame_rois'].append(
                    {
                        "roi_score": -1.0, 
                        "roi_h": h, 
                        "roi_x": x, 
                        "roi_y": y, 
                        "roi_label": {
                            "label_id": 1, 
                            "label_name": "true"
                        }, 
                        "roi_w": w, 
                        "roi_id": roiid
                    }
                )
                roiid += 1
            frameid += 1
                
                
        
    def updateInfoFromDir(self,dirname):
        dirlist = sorted(glob.glob(dirname+'/*'))
        for subdir in dirlist:
            label_name = os.path.basename(subdir)
            if not os.path.isdir(subdir):
                continue
            files = glob.glob(subdir+'/*')
            for file in files:
                file = os.path.basename(file)
                nums = re.findall(r'[\d]+',file) # 1234_4567 => ('1234','4567')
                if len(nums) != 2:
                    continue
                fnum = int(nums[0])
                roi_id = int(nums[1])
                #print(fnum,roi_id,label_name)
                ret,label_id = self.getLabelId(label_name)
                if ret:
                    rois = self.data['frames'][fnum]['frame_rois']
                    for roi in rois:
                        if roi['roi_id'] == roi_id:
                            roi['roi_label']['label_id'] = label_id
                            roi['roi_label']['label_name'] = label_name
                            #print('update: fnum=%d roi_id=%d label=%s (%d)'%
                            #      (fnum,roi_id,label_name,label_id))
            
    def createEmptyData(self,fcnt):
        data = copy.deepcopy(ImageSource.EmptyData)
        for i in range(int(fcnt)):
            frame = copy.deepcopy(ImageSource.EmptyFrame)
            frame["frame_id"] = i
            data["frames"].append(frame)
        self.data = data
        self.roisInputEnable = True
            
    def getSize(self):
        """ return image size """
        if self.videoInputEnable:
            if self.isAvi:
                if cv2ver() == 3:
                    width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
                    height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
                else:
                    width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)   # float
                    height = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) # float
            else:
                img = cv2.imread(self.videoList[0])
                height, width = img.shape[:2]
            return (int(width),int(height))
        else:
            return (640,480)

    def getFrameCount(self):
        if self.videoInputEnable:
            return self.fcnt
        else:
            return 100

    def getImage(self,fnum):
        """ return (true/false, rois) """
        if self.videoInputEnable:
            if self.isAvi:
                if cv2ver() == 3:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES,fnum)
                else:
                    self.cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,fnum)
                return self.cap.read()
            else:
                img = cv2.imread(self.videoList[fnum])
                return True,img
        return (False,[])


    def getRois(self,fnum):
        """ return (true/false, rois) """
        if self.roisInputEnable:
            try:
                js = self.data['frames'][fnum]['frame_rois']
                return (True,js)
            except:
                pass
        return (False,[])

    def saveJson(self):
        if not self.roisInputEnable:
            return
        filename,ext = os.path.splitext(self.jsonfile)
        timestamp = time.strftime("%Y%m%d%H%M%S")
        newname = filename+'_'+timestamp+ext
        with open(newname,'w') as fp:
            json.dump(self.data,fp,sort_keys=True,indent=4, separators=(',', ': '))

    def openLabel(self,filename):
        with open(filename) as fd:
            typedata = json.load(fd)
        self.labels = []
        self.labels.append(('Unsorted',-1))
        for key,value in typedata['neg_dics'].iteritems():
            self.labels.append((str(key),value))
        for key,value in typedata['pos_dics'].iteritems():
            self.labels.append((str(key),value))

    def getLabelFromData(self,data):
        frames = data['frames'] # frame list
        labels = {}
        for frame in frames:
            rois = frame['frame_rois']
            for roi in rois:
                label_id = roi['roi_label']['label_id']
                label_name = str(roi['roi_label']['label_name'])
                labels[label_name] = label_id
        return [(key,id) for key,id in labels.iteritems()]
        
    def updateRoiLabel(self):
        if not self.roisInputEnable:
            return
        for frame in self.data["frames"]:
            rois = frame["frame_rois"]
            for roi in rois:
                labelname = roi["roi_label"]["label_name"]
                ret,id = self.getLabelId(labelname)
                if ret and id >= 0:
                    roi["roi_label"]["label_id"] = id

    def getLabelNameList(self):
        return [key for key,id in self.labels]
    
    def getLabelId(self,labelname):
        for key,id in self.labels:
            if key == labelname:
                return True,id
        return False,-1

    def saveMask(self):
        if not self.videoInputEnable:
            return
        if not self.roisInputEnable:
            return
        
        # save images to a directory
        filename,ext = os.path.splitext(self.videofile)
        if os.path.exists(filename):
            if not os.path.isdir(filename):
                print('Error: ' + filename + ' is not a directory')
                return
        else:
            os.makedirs(filename)
        
        (imgw,imgh) = self.getSize()
        for fm in self.data['frames']:
            fnum = fm['frame_id']
            rois = fm['frame_rois']
            # create frame and paint rois on it
            size = (imgh,imgw,1)
            img = np.zeros(size,np.int16)
            for roi in rois:
                x0 = roi['roi_x']
                y0 = roi['roi_y']
                x1 = roi['roi_x'] + roi['roi_w'] - 1
                y1 = roi['roi_y'] + roi['roi_h'] - 1
                id = roi['roi_label']['label_id']
                cv2.rectangle(img,(x0,y0),(x1,y1),id+128,-1)
            # save frame to file
            imgname = "%s/%04d.png"%(filename,fnum)
            #print(imgname)
            cv2.imwrite(imgname,img)

    def saveVideo(self):
        if not self.videoInputEnable:
            return
        if not self.roisInputEnable:
            return
        
        # save images to a directory
        filename,ext = os.path.splitext(self.videofile)
        if os.path.exists(filename):
            if not os.path.isdir(filename):
                print('Error: ' + filename + ' is not a directory')
                return
        else:
            os.makedirs(filename)
        
        (imgw,imgh) = self.getSize()

        for fnum in range(int(self.fcnt)):
            statV,img = self.getImage(fnum)
            statR,rois = self.getRois(fnum)
            if not statV or not statR:
                continue
            for roi in rois:
                label_id = roi['roi_label']['label_id']
                if label_id <= 0:
                    continue
                color = ColorTable[getColor(label_id)]
                x0 = roi['roi_x']
                y0 = roi['roi_y']
                x1 = x0+roi['roi_w']-1
                y1 = y0+roi['roi_h']-1
                cv2.rectangle(img,(x0,y0),(x1,y1),color,1)
            # save frame to file
            imgname = "%s/img%04d.jpg"%(filename,fnum)
            print(imgname)
            cv2.imwrite(imgname,img)


    
if __name__ == "__main__":
    src = ImageSource('00161_raw.avi','objects_00161.json')
    print (src.getSize())
    print (src.getFrameCount())
    fnum = 0
    while (1):
        ret,frame,js = src.getFrame(fnum)
        fnum += 1
        if(not ret):
            break

        for roi in js:
            x0 = roi['roi_x']
            y0 = roi['roi_y']
            x1 = roi['roi_x'] + roi['roi_w'] - 1
            y1 = roi['roi_y'] + roi['roi_h'] - 1
            cv2.rectangle(frame,(x0,y0),(x1,y1),(0,256,0),1)

        cv2.imshow('frame',frame);
        cv2.waitKey(33);

    
    
        
