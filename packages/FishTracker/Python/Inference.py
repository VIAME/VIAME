#!/usr/bin/python
import numpy as np
from selective_search import *
from create_training import *
from FlowSeg import *
import sys
import os
import argparse
import glob
import json
import cv2
import caffe
import ImageIO

class Inference:
    def __init__(self, labelFile, modelFile, weightsFile, meanFile, imgW, imgH, gpu = True, mode = "net", ks = 400, opflow_thr = 40.0, flowEnable = True, removeDominateMotion = False, linearStretch = False, largeRoIRatio = 0.5, goodRoIRatio = 0.05, smallRoIRatio = 0.01):
        """mode=net,sel,flow"""
        reader = open(labelFile, "rt")
        label = json.load(reader)
        reader.close()
        self.TypeName = []
        self.TypeName.append((key for key, value in label["neg_dics"].items() if value == 0).next())
        for i in range(1, len(label["pos_dics"]) + 1):
            self.TypeName.append((key for key, value in label["pos_dics"].items() if value == i).next())

        self.width = imgW
        self.height = imgH
        self.mode = mode
        self.ks = ks
        self.flowEnable = flowEnable
	self.removeDominateMotion = removeDominateMotion
	self.linearStretch = linearStretch
        self.largeRoIRatio = largeRoIRatio
        self.goodRoIRatio = goodRoIRatio
        self.smallRoIRatio = smallRoIRatio

        if gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
        # caffe init
        self.net = caffe.Classifier(modelFile, weightsFile,
                                    mean=np.load(meanFile).mean(1).mean(1),
                                    channel_swap=(0,1,2), image_dims =(256,256),
                                    raw_scale=255)
        # flow segmentation init
        self.fs = FlowSeg()
        self.fs.init(int(self.width),int(self.height), # cols,rows
                     1, #finlev
                     3, #toplev
                     True, #laplacian
                     1, #lev_iterations
                     True) #flow_quantize
        #double opflow_thr, int min_blob_size, int max_blob_size)
        self.fs.params(opflow_thr, 1000, int(self.width*self.height/4), removeDominateMotion) 

        """
        self.ColorTbl = (
            (0  ,255,0),  # 0: green
            (255,0  ,0),  # 1: blue
            (0  ,255,255),# 2: yellow
            (255,255,0),  # 3: Cyan
            (255,0  ,255),# 4: Magenta
            (0  ,0  ,80), # 5: Maroon
            (0  ,80 ,80), # 6: Olive
            (80 ,0  ,80), # 7: purple
            (80 ,80 ,0),  # 8: Teal
            (80 ,0  ,0),  # 9: Navy
            (255,255,255),# 10: white 
            (0  ,0  ,0),  # 11: black
            (128,128,128) # gray    
            )
        """
        self.ColorTbl = (
            (  0,  0,  0),# 0: black
            (  0,  0,128),# 1: maroon
            (  0,255,  0),# 2: green
            (  0,128,128),# 3: olive
            (128,  0,  0),# 4: navy
            (128,  0,128),# 5: purple
            (128,128,  0),# 6: teal
            (192,192,192),# 7: silver
            (128,128,128),# 8: gray    
            (  0,  0,255),# 9: red    
            (  0,255,191),# 10: lime    
            (  0,255,255),# 11: yellow
            (255,  0,  0),# 12: blue
            (255,  0,255),# 13: fuchsia
            (255,255,  0),# 14: aqua
            (255,255,255),# 15: white 
            (  0,  0,  0),# 16: black
            (  0,  0,128),# 17: maroon
            (  0,255,  0),# 18: green
            (  0,128,128),# 19: olive
            (128,  0,  0),# 20: navy
            (128,  0,128),# 21: purple
            (128,128,  0),# 22: teal
            (192,192,192),# 23: silver
            (128,128,128),# 24: gray    
            (  0,  0,255),# 25: red    
            (  0,255,191),# 26: lime    
            (  0,255,255),# 27: yellow
            (255,  0,  0),# 28: blue
            (255,  0,255),# 29: fuchsia
            (255,255,  0),# 30: aqua
            (255,255,255) # 31: white 
            )
        print "Inference init done"


    def execute(self,img,fnum):
        # run selective search
        rois = []
        scores = np.empty(shape = (0,))
        stretch_img = img
        if self.linearStretch:
            # linear stretching the input image
            area = self.width * self.height
            hist = cv2.calcHist([img], [0], None,[256],[0,256])
            low_thresh = -1
            high_thresh = -1
            hist_cutoff = 0.001
            accumu = 0.0
            for i in range(256):
                accumu = accumu + hist[i] / area
                if accumu < hist_cutoff :
                    low_thresh = i
                elif accumu < 1.0 - hist_cutoff:
                    high_thresh = i
            cv2.convertScaleAbs(img, stretch_img, 255.0/(high_thresh-low_thresh), -255.0/(high_thresh-low_thresh))
            img = stretch_img

        if( self.mode == "sel" or self.mode == "net"):
            selRois = self.selSearch(stretch_img)
            #print "ssLen=%d"%(len(selRois))
        if( self.mode == "sel" and not self.flowEnable):
            return (selRois, scores)

        if self.flowEnable:
            # run flow segmentation
            if( self.mode == "flow"):
                (negRois, posRois) = self.flowSegment(img, fnum, debugShow = True)
            if( self.mode == "sel" or self.mode == "net"):
                (negRois, posRois) = self.flowSegment(img, fnum)
                #print "flowLen=%d"%(len(posRois))
            if( self.mode == "flow"):
                return (posRois, scores)
            # put two lists together
            rois = selRois + posRois
        elif self.mode == "flow":
            return (rois, scores)
        else:
            rois = selRois 

        # clean up list
        rois = removeDuplicates(rois, 0.95)
        rois = removeInvalid(rois, self.width, self.height)
        rois = removeSize(rois, self.width, self.height, self.largeRoIRatio, self.smallRoIRatio)
        #print "listLen=%d"%(len(rois))
        # create list of images
        roiImgs = []
        if( self.mode == "net"):
            for roi in rois:
                roiImg = ImageIO.crop(img,roi,border=30)
                roiImg = skimage.img_as_float(roiImg).astype(np.float32)
                roiImgs.append(roiImg)
        # call caffe
        if roiImgs:
            scores = self.net.predict(roiImgs, oversample = False)
        # print scores
        # return scores for each roi
    	#roisWithScores = zip(selRois,scores)
        return (rois, scores)

    def selSearch(self,img):
        """ return list of (x,y,w,h)"""
        imgPyr = cv2.pyrDown(img)
 
        # selective search
        regions = selective_search(imgPyr,color_spaces = ["rgb"], 
                                   ks = [self.ks],
                                   feature_masks = [features.SimilarityMask
                                                   (size = True, color = True,
                                                   texture = True, fill = True)])
        selRois = [(x1, y1, x2-x1, y2-y1) for v, (y1, x1, y2, x2) in regions]
        selRois = [(2*x, 2*y, 2*w, 2*h) for (x, y, w, h) in selRois]
        return selRois

    def flowSegment(self, img, fnum, debugShow = False):
        """return (negRois:x,y,w,h,posRois:x,y,w,h) """
        return self.fs.run(img, fnum, debugShow) 

    def paint(self,img,rois,scores):
        if(self.mode == "sel" or self.mode == "flow" or scores.size == 0):
            return self.box_paint(img,rois,scores)
        #return self.nms_paint(img,rois,scores)
        #return self.type_paint(img,rois,scores)
        return self.nms_paint3(img,rois,scores)

    def box_paint(self,img,rois,scores):
        roiarray = []
        roi_id = 0
        for (x,y,w,h) in rois:
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 1)
            if w > self.goodRoIRatio * self.width and h > self.goodRoIRatio * self.height:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, "{0}:".format(roi_id), (int(x + 3), int(y + 3)), font, 0.5, (255, 255, 255), 1)

                label = dict()
                label["label_id"] = -1
                label["label_name"] = "Unsorted"
                roi = dict()
                roi["roi_id"] = roi_id
                roi["roi_x"] = x
                roi["roi_y"] = y
                roi["roi_w"] = w
                roi["roi_h"] = h
                roi["roi_score"] = -1.0
                roi["roi_label"] = label
                roiarray.append(roi)
                roi_id += 1
        return img, roiarray

    def type_paint(self,img,rois,scores):
        """ paint boxed using different color for each type"""
        scored_boxes = []
        lens = len(rois)  
        for i in range(lens):
            (x,y,w,h) =  rois[i]
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            if(maxIdx[0] != 0):
                print scores[i]
                print "maxIdx=%d"%(maxIdx[0])
            maxScore = scores[i][maxIdx[0]]
            if maxIdx[0]<len(self.ColorTbl):
                thickness = 1
                if maxIdx[0] > 0:
                    thickness = 3
            cv2.rectangle(img,(x,y),(x+w-1,y+h-1),self.ColorTbl[maxIdx[0]],thickness)
        return img

    def nms_paint(self,img,rois,scores):
        scored_boxes = []
        lens = len(rois)  
        for i in range(lens):
            (x,y,w,h) =  rois[i]
            pos_score = scores[i][1]
            if pos_score > -0.5:
                scored_box = [x, y, x+w-1, y+h-1, pos_score]
                scored_boxes.append(scored_box)
                #cv2.rectangle(img,(x,y),(x+w-1,y+h-1),(0,128,0),1)
    
        nms_boxes = self.nms(np.array(scored_boxes), 0.1)
        #print nms_boxes
    
        lens = len(nms_boxes) 
        for i in range(lens):
            score = nms_boxes[i][4]
            x0 = int(nms_boxes[i][0])
            y0 = int(nms_boxes[i][1])
            x1 = int(nms_boxes[i][2])
            y1 = int(nms_boxes[i][3])
            if score > 0:
                cv2.rectangle(img,(x0,y0),(x1,y1),(0,0,255),1)
        return img

    def nms_paint3(self,img,rois,scores):
        nmsBoxes = self.nmsEx3(rois, scores, 0.1)
        roiarray = []
        roi_id = 0
        for (x0, y0, x1, y1, t, score) in nmsBoxes:
            if x1 - x0 > self.smallRoIRatio * self.width and y1 - y0 > self.smallRoIRatio * self.height:
                thickness = 1
                if t > 0:
                    thickness = 3
                cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), self.ColorTbl[int(t)], thickness)
                if t > 0:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img,"{0}: {1}".format(int(t), self.TypeName[int(t)]), (int(x0+3), int(y0+3)), font, 0.5, (255, 255, 255), 1)

                label = dict()
                label["label_id"] = int(t)
                label["label_name"] = self.TypeName[int(t)]
                roi = dict()
                roi["roi_id"] = roi_id
                roi["roi_x"] = int(x0)
                roi["roi_y"] = int(y0)
                roi["roi_w"] = int(x1) - int(x0) + 1
                roi["roi_h"] = int(y1) - int(y0) + 1
                roi["roi_score"] = score
                roi["roi_label"] = label
                roiarray.append(roi)
                roi_id += 1
        return img, roiarray

    def nms_paint2(self,img,rois,scores):
        nmsBoxes = self.nmsEx2(rois,scores,0.1)
        typeLens = len(nmsBoxes)
        for t in range(typeLens):
            for (x0,y0,x1,y1,score) in nmsBoxes[t]:
                thickness = 1
                if t > 0:
                    thickness = 3
                cv2.rectangle(img,(int(x0),int(y0)),(int(x1),int(y1)),self.ColorTbl[t],thickness)
        return img

    def nmsEx3(self,rois,scores,overlap):
        """ run nms on whole list 
        rois:[(x,y,w,h),...]
        scores:[(...),...]  each roi has score for each type
        return: [(x,y,w,h,type,score),...]
        """
        lens = len(rois) # lens of rois and scores must be same
        # high score index for each roi
        typeIdx = []
        scored_boxes = []
        for i in range(lens):
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            typeIdx.append(maxIdx[0])
            (x,y,w,h) =  rois[i]
            score = scores[i][typeIdx[i]]
            scored_box = [x, y, x+w-1, y+h-1,typeIdx[i],score]
            scored_boxes.append(scored_box)
        return self.nms(np.array(scored_boxes), overlap)

    def nmsEx2(self,rois,scores,overlap):
        """
        run nms on each type
        rois:[(x,y,w,h),...]
        scores:[(...),...]  each roi has score for each type
        return:[[(x,y,x1,y1,score),...],...] scored rois for each type
        """
        lens = len(rois) # lens of rois and scores must be same
        # high score index for each roi
        typeIdx = []
        for i in range(lens):
            maxIdx = numpy.unravel_index(scores[i].argmax(),scores[i].shape)
            typeIdx.append(maxIdx[0])
        # nms for each type
        typeLen = len(scores[0])
        retVal = []
        for t in range(typeLen):
            # build score box for each type
            scored_boxes = []
            for i in range(lens):
                if typeIdx[i] != t:
                    continue
                (x,y,w,h) =  rois[i]
                score = scores[i][typeIdx[i]]
            scored_box = [x, y, x+w-1, y+h-1, score]
            scored_boxes.append(scored_box)
            nms_boxes = self.nms(np.array(scored_boxes), overlap)
            retVal.append(nms_boxes)
        return retVal

    def nms(self,boxes, overlap):
        """
        original code: http://github.com/quantombone/exemplarsvm/internal/esvm_nms.m
        boxes: [(x,y,x1,y1,...,score),...]
        return:[(x,y,x1,y1,...,score),...]
        """
        if boxes.size==0: 
            return []
    
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        s = boxes[:,-1]
    
        area = (x2-x1+1) * (y2-y1+1)

        I = np.argsort(s)
 
        pick = np.zeros(s.size, dtype=np.int)
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
            
            I = np.delete(I, np.concatenate([[last], np.nonzero(o > overlap)[0]]))
        
        pick = pick[:counter]
        top = boxes[pick,:]
        return top

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="input video file/image folder"
    )
    # Optional arguments.
    parser.add_argument(
        "--start",
        default="0",
        help="start frame#"
    )
    parser.add_argument(
        "--stop",
        default="1000000",
        help="stop frame#"
    )
    parser.add_argument(
        "--gpu",
        default="False",
        help="gpu enable: True|False"
    )
    parser.add_argument(
        "--mode",
        default="net",
        help="mode=net|sel|flow"
    )
    parser.add_argument(
        "--ks",
        default="200",
        help="ks in selective search"
    )
    parser.add_argument(
        "--flowthr",
        default="10.0",
        help="opflow_thr in flowSeg"
    )
    parser.add_argument(
        "--flowEnable",
        default="True",
        help="flow seg enable"
    )
    parser.add_argument(
        "--model_type",
        default="model/mbari_type.json",
        help="Model type labels (.json)"
    )
    parser.add_argument(
        "--model_def",
        default="model/mbari_type_deploy.prototxt",
        help="Model definition file (.prototxt)"
    )
    parser.add_argument(
        "--pretrained_model",
        default="model/mbari_type_iter_80000.caffemodel",
        help="Trained model weights file(.caffemodel)."
    )
    parser.add_argument(
        "--mean_file",
        default="model/ilsvrc_2012_mean.npy",
        help="mean file (.npy)"
    )
    parser.add_argument(
        "--removeDominateMotion",
        default="False",
        help="Remove dominating motion caused by camera"
    )
    parser.add_argument(
        "--linearStretch",
        default="False",
        help="Linear stretching the input image"
    )
    parser.add_argument(
        "--largeRoIRatio",
        default="0.5",
        help="Linear stretching the input image"
    )
    parser.add_argument(
        "--goodRoIRatio",
        default="0.05",
        help="Linear stretching the input image"
    )
    parser.add_argument(
        "--smallRoIRatio",
        default="0.01",
        help="Linear stretching the input image"
    )
    args = parser.parse_args()

    start_frame = int(args.start)
    stop_frame = int(args.stop)
    if os.path.isfile(os.path.join(os.environ["FS_ROOT"], args.input_file)):
        cap = cv2.VideoCapture(os.path.join(os.environ["FS_ROOT"], args.input_file))
        if cap.isOpened() == False:
            print "Cannot open input file:%s"%(os.path.join(os.environ["FS_ROOT"], args.input_file))
            exit(1)
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps    = cap.get(cv2.CAP_PROP_FPS)
        fnums  = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    else:
        filelist = sorted(glob.glob(os.path.join(os.environ["FS_ROOT"], args.input_file, "*.jpg")))
        fnums = len(filelist)
        img = cv2.imread(filelist[0])
        height, width = img.shape[:2]
    stop_frame = min(stop_frame, int(fnums))

    # create inferencing object
    inferencing = Inference(os.path.join(os.environ["FS_ROOT"], args.model_type), 
                            os.path.join(os.environ["FS_ROOT"], args.model_def),
                            os.path.join(os.environ["FS_ROOT"], args.pretrained_model),
                            os.path.join(os.environ["FS_ROOT"], args.mean_file),
                            width,
                            height,
                            gpu = (args.gpu == "True" or args.gpu == "true"),
                            mode = args.mode,
                            ks = int(args.ks),
                            opflow_thr = float(args.flowthr),
                            flowEnable = (args.flowEnable == "True" or args.flowEnable == "true"),
                            removeDominateMotion = (args.removeDominateMotion == "True" or args.removeDominateMotion == "true"),
                            linearStretch = (args.linearStretch == "True" or args.linearStretch == "true"),
                            largeRoIRatio = float(args.largeRoIRatio),
                            goodRoIRatio = float(args.goodRoIRatio),
                            smallRoIRatio = float(args.smallRoIRatio))

    cv2.namedWindow("Source", cv2.WINDOW_AUTOSIZE);
    frames = []
    sel_frames = []
    for f in xrange(start_frame, stop_frame):
        if os.path.isfile(os.path.join(os.environ["FS_ROOT"], args.input_file)):
            ret,img = cap.read() # read image
        else:
            img = cv2.imread(filelist[f])
            ret = img is not None
 
        if ret:
            (selRois,scores) = inferencing.execute(img, f - start_frame)
            # Non Maximum Suppression
            #print selRois
            #print scores
            img, rois = inferencing.paint(img, selRois, scores)
            frame = dict()
            frame["frame_id"] = f
            frame["frame_rois"] = rois
            #print json.dumps(frame, indent = 4)
            frames.append(frame)
            sel_frames.append(frame)

            clip = f / 900;
            if args.mode == "flow" and (args.flowEnable == "False" or args.flowEnable == "false"):
                 outDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "clip{0:03d}".format(clip), "raw")
                 selDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "selected", "raw")

            elif args.mode == "sel" and (args.flowEnable == "True" or args.flowEnable == "true"):
                 outDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "clip{0:03d}".format(clip), "prop")
                 selDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "selected", "prop")
            else:
                 outDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "clip{0:03d}".format(clip), "class")
                 selDir = os.path.join(os.environ["FS_ROOT"], "video", os.path.splitext(os.path.basename(args.input_file))[0], "selected", "class")
            outFile = "%s/images/%06d.jpg" % (outDir, f)
            print outFile
            ImageIO.saveImg(outFile, img)
            outFile = "%s/images/%06d.jpg" % (selDir, f)
            ImageIO.saveImg(outFile, img)

            if (f + 1) / 900 > clip:
                data = dict()
                data["input_file"] = args.input_file
                data["frames"] = frames
                frames = []
                writer = open(os.path.join(outDir, "objects.json"), "wt")
                writer.write(json.dumps(data, indent = 4))
                writer.close()
                print "Inference results saved"

            cv2.imshow("Source", img)
            cv2.waitKey(100)
        else:
            print "Cannot read Frame #:{0}".format(f)
    if frames:
        data = dict()
        data["input_file"] = args.input_file
        data["frames"] = frames
        writer = open(os.path.join(outDir, "objects.json"), "wt")
        writer.write(json.dumps(data, indent = 4))
        writer.close()
        print "Inference results saved"
    data = dict()
    data["input_file"] = args.input_file
    data["frames"] = sel_frames
    writer = open(os.path.join(selDir, "objects.json"), "wt")
    writer.write(json.dumps(data, indent = 4))
    writer.close()

if __name__ == "__main__":
    main(sys.argv)

