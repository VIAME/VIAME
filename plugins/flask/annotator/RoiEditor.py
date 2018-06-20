#! /usr/bin/env python
import copy

class RoiEditor:
    NotTouched = 0
    Touched = 1
    
    TopLeft = 2
    Top = 3
    TopRight = 4
    Left = 5
    Center = 6
    Right = 7
    BottomLeft = 8
    Bottom = 9
    BottomRight = 10

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

    def __init__(self,rois,imgsize):
        self.rois = rois
        self.imgSize = imgsize
        self.dragged = False
        self.addRoiState = False
        self.zoomState = False
        self.addVertexState = False
        self.zoomRoi = [0,0,0,0] # (x0,y0,x1,y1)
        self.nextId = -1
        for roi in self.rois:
            if self.nextId < roi['roi_id']:
                self.nextId = roi['roi_id']
        self.nextId += 1
        self.fixRois(rois)
                
    def touched(self,pos,edge=4):
        """return roiId"""
        all = []
        for roi in self.rois:
            id = roi['roi_id']
            x = roi['roi_x']
            y = roi['roi_y']
            w = roi['roi_w']
            h = roi['roi_h']
            if pos[0]<x-edge or pos[0]>x+w+edge or pos[1]<y-edge or pos[1]>y+h+edge:
                continue
            touch = RoiEditor.Center
            if pos[0] < x+edge:
                touch = RoiEditor.Left
            elif pos[0] > x+w-edge:
                touch = RoiEditor.Right
            if pos[1] < y+edge:
                if touch == RoiEditor.Right:
                    touch = RoiEditor.TopRight
                elif touch == RoiEditor.Left:
                    touch = RoiEditor.TopLeft
                else:
                    touch = RoiEditor.Top
            if pos[1] > y+h-edge:
                if touch == RoiEditor.Right:
                    touch = RoiEditor.BottomRight
                elif touch == RoiEditor.Left:
                    touch = RoiEditor.BottomLeft
                else:
                    touch = RoiEditor.Bottom
            all.append((id,touch))
        if len(all) == 0:
            return (-1,RoiEditor.NotTouched)
        elif len(all) > 1:
            for id, touch in all:
                if touch != RoiEditor.Center:
                    return (id,touch)
        return all[0]

    def vertexTouched(self,pos,selId,edge = 8):
        if selId <0:
            return selId,-1
        for roi in self.rois:
            if selId != roi['roi_id']:
                continue
            if not 'vertices' in roi:
                continue
            for idx,vertex in enumerate(roi['vertices']):
                if pos[0] > vertex[0] + edge:
                    continue
                if pos[0] < vertex[0] - edge:
                    continue
                if pos[1] > vertex[1] + edge:
                    continue
                if pos[1] < vertex[1] - edge:
                    continue
                return selId,idx
        return selId,-1
        

    def setValues(self,roiInfo,id):
        for roi in self.rois:
            if id == roi['roi_id']:
                for key,edit in roiInfo.iteritems():
                    if key == 'label_name':
                        roi['roi_label'][key] = str(edit.text())
                    elif key == 'label_id':
                        roi['roi_label'][key] = int(edit.text())
                    elif key == 'roi_score':
                        roi[key] = float(edit.text())
                    else:
                        roi[key] = int(edit.text())

    def delRoi(self,id):
        for roi in self.rois:
            if id == roi['roi_id']:
                self.rois.remove(roi)

    def delRois(self,zoomed,selId):
        """ delete all expcept selected """
        delList = []
        for roi in self.rois:
            if zoomed:
                x = roi['roi_x']
                y = roi['roi_y']
                w = roi['roi_w']
                h = roi['roi_h']
                if x < self.zoomRoi[0] or y <self.zoomRoi[1]:
                    continue
                if x+w > self.zoomRoi[2] or y+h > self.zoomRoi[3]:
                    continue
            if selId < 0 or selId != roi['roi_id']:
                delList.append(roi['roi_id'])
        for id in delList:
            self.delRoi(id)

    def setAddRoiState(self,state):
        self.addRoiState = state

    def setAddVertexState(self,state,selId):
        if selId < 0:
            return
        self.addVertexState = state

    def addVertex(self,pos,selId):
        if selId < 0:
            return
        for roi in self.rois:
            if selId == roi['roi_id']:
                if not 'vertices' in roi:
                    roi['vertices'] = []
                insIdx = self.minDisIdx(roi['vertices'],pos)
                if insIdx >= 0:
                    roi['vertices'].insert(insIdx,pos)

    def minDisIdx(self,vertices,pos):
        """return idx with same slope"""
        vl = len(vertices)
        if vl < 3:
            return vl+1
        minSlope = 0
        minIdx = -1;
        for idx,vertex in enumerate(vertices):
            if idx == 0:
                prev = vertices[vl-1]
            else:
                prev = vertices[idx-1]

            ds = (((pos[1]-vertex[1])*(pos[1]-vertex[1]) +
                   (pos[0]-vertex[0])*(pos[0]-vertex[0])) + 
                  ((pos[1]-prev[1])*(pos[1]-prev[1]) +
                   (pos[0]-prev[0])*(pos[0]-prev[0])))

            if idx == 0 or minSlope > ds:
                minSlope = ds
                minIdx = idx
        return minIdx


    def changeX(self,v,selId):
        for roi in self.rois:
            if selId == roi['roi_id']:
                roi['roi_x'] += v
                
    def changeY(self,v,selId):
        for roi in self.rois:
            if selId == roi['roi_id']:
                roi['roi_y'] += v

    def changeH(self,v,selId):
        for roi in self.rois:
            if selId == roi['roi_id']:
                roi['roi_h'] += v
                
    def changeW(self,v,selId):
        for roi in self.rois:
            if selId == roi['roi_id']:
                roi['roi_w'] += v
    
    def dragging(self,pos,selId):
        if not self.dragged and len(pos)>0:
            if self.addRoiState:
                # create new roi
                roi = copy.deepcopy(RoiEditor.EmptyRoi)
                roi['roi_id'] = self.nextId
                roi['roi_x'] = pos[0]
                roi['roi_y'] = pos[1]
                id,touch = self.nextId,RoiEditor.BottomRight
                selId = id # change selId to new created roi
                self.nextId += 1
                self.rois.append(roi)
            elif self.addVertexState:
                id,touch = self.vertexTouched(pos,selId)
            else:
                id,touch = self.touched(pos)

            if id != selId or selId < 0 :
                # no selected, dragging zoomRoi
                self.zoomState = True
                self.zoomRoi = [pos[0],pos[1],pos[0],pos[1]] # (x0,y0,x1,y1)
                self.dragged = True
                return False,selId 

            # now mouse touched selected ROI
            for roi in self.rois:
                if id == roi['roi_id']:
                    self.roi = roi
                    self.x = roi['roi_x']
                    self.y = roi['roi_y']
                    self.w = roi['roi_w']
                    self.h = roi['roi_h']
                    self.touch = touch
                    self.pos = pos
                    self.dragged = True
            return False,selId
        elif self.dragged and len(pos) == 0:
            # stop
            if self.addVertexState:
                self.validateVertex()
            self.addRoiState = False
            self.dragged = False
            return True,selId

        if self.dragged:
            if self.zoomState:
                # dragging zoomRoi 
                self.zoomRoi[2] = pos[0] # x1
                self.zoomRoi[3] = pos[1] # y1
                return True,selId
            elif self.addVertexState:
                self.dragVertex(pos)
            else:
                # dragging selected ROI
                self.dragRoi(pos)
        return True,selId

    def setZoomStateOff(self):
        self.zoomState = False
        if self.zoomRoi[0] > self.zoomRoi[2]:
            temp = self.zoomRoi[0]
            self.zoomRoi[0] = self.zoomRoi[2]
            self.zoomRoi[2] = temp
        if self.zoomRoi[1] > self.zoomRoi[3]:
            temp = self.zoomRoi[1]
            self.zoomRoi[1] = self.zoomRoi[3]
            self.zoomRoi[3] = temp


    def dragRoi(self,pos):
        x = pos[0]
        y = pos[1]
        dx = pos[0] - self.pos[0]
        dy = pos[1] - self.pos[1]
        if self.touch == RoiEditor.TopLeft:
            self.roi['roi_x'] = self.x + dx
            self.roi['roi_y'] = self.y + dy
            self.roi['roi_w'] = self.w - dx
            self.roi['roi_h'] = self.h - dy
        elif self.touch == RoiEditor.Top:
            self.roi['roi_y'] = self.y + dy
            self.roi['roi_h'] = self.h - dy
        elif self.touch == RoiEditor.TopRight:
            self.roi['roi_y'] = self.y + dy
            self.roi['roi_w'] = self.w + dx
            self.roi['roi_h'] = self.h - dy
        elif self.touch == RoiEditor.Left:
            self.roi['roi_x'] = self.x + dx
            self.roi['roi_w'] = self.w - dx
        elif self.touch == RoiEditor.Center:
            self.roi['roi_x'] = self.x + dx
            self.roi['roi_y'] = self.y + dy
        elif self.touch == RoiEditor.Right:
            self.roi['roi_w'] = self.w + dx
        elif self.touch == RoiEditor.BottomLeft:
            self.roi['roi_x'] = self.x + dx
            self.roi['roi_w'] = self.w - dx
            self.roi['roi_h'] = self.h + dy
        elif self.touch == RoiEditor.Bottom:
            self.roi['roi_h'] = self.h + dy
        elif self.touch == RoiEditor.BottomRight:
            self.roi['roi_w'] = self.w + dx
            self.roi['roi_h'] = self.h + dy
        self.validateRoi(self.roi)


    def dragVertex(self,pos):
        if not 'vertices' in self.roi:
            return
        if self.touch < 0 or self.touch > len(self.roi['vertices']):
            return
        self.roi['vertices'][self.touch] = pos

    def validateVertex(self):
        if not 'vertices' in self.roi:
            return
        if self.touch < 0 or self.touch > len(self.roi['vertices']):
            return
        vertex = self.roi['vertices'][self.touch]
        (x,y,w,h) = (self.roi['roi_x'],self.roi['roi_y'],
                     self.roi['roi_w'],self.roi['roi_h'])
        if vertex[0] < x or vertex[0] > x+w or vertex[1] < y or vertex[1] > y+h:
            del(self.roi['vertices'][self.touch])

    def validateRoi(self,roi):
        """ only check roi """
        if roi['roi_x'] < 0:
            roi['roi_x'] = 0
        elif roi['roi_x'] > self.imgSize[0]-1:
            roi['roi_x'] = self.imgSize[0]-1

        if roi['roi_y'] < 0:
            roi['roi_y'] = 0
        elif roi['roi_y'] > self.imgSize[1]-1:
            roi['roi_y'] = self.imgSize[1]-1

        if roi['roi_w'] < 0:
            roi['roi_w'] = 0
        elif roi['roi_x'] + roi['roi_w'] > self.imgSize[0]-1:
            roi['roi_w'] = self.imgSize[0] - roi['roi_x'] -1

        if roi['roi_h'] < 0:
            roi['roi_h'] = 0
        elif roi['roi_y'] + roi['roi_h'] > self.imgSize[1]-1:
            roi['roi_h'] = self.imgSize[1] - roi['roi_y'] -1

    def fixRois(self,rois):
        """fix roi with w<0 or h<0 """
        for roi in rois:
            x = roi['roi_x']
            y = roi['roi_y']
            w = roi['roi_w']
            h = roi['roi_h']
            if w < 0:
                w = -w
                x -= w
            if h < 0:
                h = -h
                y -= h

