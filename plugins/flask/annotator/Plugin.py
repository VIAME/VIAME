import tmplMatch.tmplMatch as tm
import copy

class Plugin:
    EmptyRoi = {
        "roi_id": 0,
        "roi_score": -1.0, 
        "roi_x": 0, 
        "roi_y": 0, 
        "roi_w": 1, 
        "roi_h": 1, 
        "roi_label": {
            "label_id": -1, 
            "label_name": "Unsorted"
        }
    }

    def __init__(self,imgsrc):
        self.src = imgsrc
        self.tmplMatchEnable = False

    def setTmplMatchEnable(self,enable):
        self.tmplMatchEnable = enable

    def beginProcess(self,fnum):
        if self.tmplMatchEnable:
            return self.tmplMatchPlugIn(fnum)
        else:
            return self.noPlugIn(fnum)
        
    def noPlugIn(self,fnum):
        ret1,img = self.src.getImage(fnum)
        ret2,rois =self.src.getRois(fnum)
        return (ret1,img),(ret2,rois)

    def tmplMatchPlugIn(self,fnum):
        print('begin');
        if fnum == 0:
            return self.noPlugIn(fnum)

        ret1,img_ins  = self.src.getImage(fnum)
        ret2,rois_ins = self.src.getRois(fnum)
        ret3,img_ref  = self.src.getImage(fnum-1)
        ret4,rois_ref = self.src.getRois(fnum-1)
        if not ret1 or not ret2 or not ret3 or not ret4:
            return (ret1,img_ins),(ret2,rois_ins)

        #remove all rois in current frame
        # we may merge them later
        del rois_ins[:]

        # rois structure
        (w,h) = self.src.getSize()
        for roi in rois_ref:
            id = roi['roi_id']
            ref_rect = (roi['roi_x'],roi['roi_y'],roi['roi_w'],roi['roi_h'])
            label_id = roi['roi_label']['label_id']
            label_name = roi['roi_label']['label_name']
            ret,ins_rect = tm.tmplMatch(img_ins,img_ref,ref_rect,w,h)

            newRoi = copy.deepcopy(Plugin.EmptyRoi)
            newRoi['roi_id'] = id
            newRoi['roi_x'] = ins_rect[0]
            newRoi['roi_y'] = ins_rect[1]
            newRoi['roi_w'] = ins_rect[2]
            newRoi['roi_h'] = ins_rect[3]
            newRoi['roi_label']['label_id'] = label_id
            newRoi['roi_label']['label_name'] = label_name
            rois_ins.append(newRoi)
        return (ret1,img_ins),(ret2,rois_ins)
