from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from PIL import Image

from torch.autograd import Variable
import numpy as np

from vital.types import BoundingBox
from vital.types import DetectedObject
from vital.types import DetectedObjectSet

from kwiver.arrows.pytorch.seg_utils import *

try:
    import cv2
except ImportError:
    cv2 = None

class_names = np.array([
            'background',
            'aeroplane',
            'bicycle',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'diningtable',
            'dog',
            'horse',
            'motorbike',
            'person',
            'potted plant',
            'sheep',
            'sofa',
            'train',
            'tv/monitor',
        ])

class FCN_Segmentation(object):

    def __init__(self, model, cuda=True):
        self.cuda = cuda
        self.model = model

    def __call__(self, in_img, fcn_flag=True):
        if fcn_flag: 
            return self._apply_fcn(in_img)
        else:
            return self._apply_contour(in_img)

    def _apply_fcn(self, in_img):

        self.model.eval()

        img = transform(in_img)
        if self.cuda:
            img = img.cuda()
        v_img = Variable(img[None], volatile=True)
        score = self.model(v_img)

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

        lbl_pred_overlap = label2rgb(lbl_pred, img=in_img, n_labels=21, label_names=class_names)
        lbl_pred = label2rgb(lbl_pred, n_labels=21)

        # get bounding boxs
        cv_pred = cv2.cvtColor(np.array(lbl_pred.squeeze()), cv2.COLOR_RGB2BGR)
        lbl_pred_overlap = cv2.cvtColor(np.array(lbl_pred_overlap.squeeze()), cv2.COLOR_RGB2BGR)
        imgray = cv2.cvtColor(cv_pred, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(cv_pred, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(in_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        dos = DetectedObjectSet()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            dobj = DetectedObject(bbox=BoundingBox(float(x), float(y), float(x + w), float(y + h)), confid=1.0)
            dos.add(dobj)

        return dos, in_img, lbl_pred_overlap

    def _apply_contour(self, in_img):

        # get bounding boxs
        cv_img = cv2.cvtColor(np.array(in_img), cv2.COLOR_RGB2BGR)
        imgray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(imgray, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(cv_pred, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.rectangle(in_img, (x, y), (x + w, y + h), (255, 255, 255), 2)

        dos = DetectedObjectSet()
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            dobj = DetectedObject(bbox=BoundingBox(float(x), float(y), float(x + w), float(y + h)), confid=1.0)
            dos.add(dobj)

        return dos, in_img, None
