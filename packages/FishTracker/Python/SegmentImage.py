# SegmentImage class to run selective_search on an image

from selective_search import *
from timeit import default_timer as timer
import cv2
import traceback

class SegmentImage :

    def __init__(self, width, height, pyr_levels = 1, monochrome = True) :
        self.segment_time = -1
        if monochrome :
            self.color = False
        else :
            self.color = True
        self.width = width
        self.height = height
        self.pyr_levels = max(1, pyr_levels)
        self.setBorder()


    def segmentFrame(self, img1, img0 = None, img2 = None) :
        """Find object ROIS in a frame using multiple pyramid levels.

            img1 image at t=0 for color, or monochrome image
            img0 image at -t for color
            img2 image at +t for color
            returns ok, [(v, (y0, x0, y1, x1))]
            where
              ok          True if there was no error
              v           random value returned by selective_search
              (x0, y0)    coordinate of one corner of the ROI
              (x1, y1)    coordinate of the opposite corner of the ROI
        """
        start = timer()
        self.segment_time = -1
        if self.color :
            img = cv2.merge((img0[:, :, 0], img1[:, :, 0], img2[:, :, 0]))
        else :
            img = img1
        rois = []
        return_ok = True
        for pyr_level in range(self.pyr_levels) :
            if pyr_level == 0 :
                img_tmp = img.copy()
            else :
                img_tmp = cv2.pyrDown(img_pyr)
            img_pyr = img_tmp
            ret, pyr_rois = self.segmentPyrLevel(img_pyr, pyr_level)
            if ret :
                #debug pyr_rois.sort(key = lambda roi : roi[1])
                if (pyr_level > 0) :
                    pyr_rois = self.removeDuplicateRois(rois, pyr_rois, 3 << pyr_level)
                rois = rois + pyr_rois
            else :
                print '***Error in segmentPyrLevel at pyr level', pyr_level
                return_ok = False

        self.segment_time = timer() - start
        return return_ok, rois


    def segmentPyrLevel(self, img, pyr_level = 0) :
        """Find object ROIS in a frame at a single pyramid level"""
        rois = []
        try :
            regions = selective_search(img,
                color_spaces = ['lab'],
                ks = [500],
                feature_masks = [features.SimilarityMask(size = True,
                        color = self.color,
                        texture = True,
                        fill = True)])

            # create a list of ROIs at pyramid level 0
            factor = 1 << pyr_level
            for i in range(len(regions)) :
                # regions is [(v,(y0, x0, y1, x1))]
                y0 = factor * regions[i][1][0]
                if y0 < self.minY : continue
                if y0 > self.maxY : continue

                x0 = factor * regions[i][1][1]
                if x0 < self.minX : continue
                if x0 > self.maxX : continue

                y1 = factor * regions[i][1][2]
                if y1 < self.minY : continue
                if y1 > self.maxY : continue

                x1 = factor * regions[i][1][3]
                if x1 < self.minX : continue
                if x1 > self.maxX : continue

                if x0 == x1 : continue
                if y0 == y1 : continue
                
                roi = tuple((regions[i][0], tuple((y0, x0, y1, x1))))
                rois.append(roi)
            ret = True

        except ValueError :
            traceback.print_exc()
            ret = False

        return ret, rois


    def setBorder(self, border = 1) :
        self.border = border
        self.minX = border
        self.maxX = self.width - border - 1
        self.minY = border
        self.maxY = self.height - border - 1


    def drawRois(self, img, rois, bgr = (0, 255, 0), line_size = 1) :
        for v, (y0, x0, y1, x1) in rois:
            cv2.rectangle(img, (x0, y0), (x1, y1), bgr, line_size)


    def removeDuplicateRois(self, old_rois, new_rois, max_difference) :
        unique_rois = []
        for new_roi in new_rois :
            if not self.isDuplicateRoi(new_roi, old_rois, max_difference) :
                unique_rois.append(new_roi)
        #debug print "removeDuplicateRois", len(new_rois), len(unique_rois)
        return unique_rois


    def isDuplicateRoi(self, new_roi, old_rois, max_difference) :
        for old_roi in old_rois :
            result = True
            for i in range(len(new_roi)) :
                if abs(new_roi[1][i] - old_roi[1][i]) > max_difference :
                    result = False
                    break
            if result :
                #debug print 'duplicate', max_difference, new_roi, old_roi
                return True
        return False
