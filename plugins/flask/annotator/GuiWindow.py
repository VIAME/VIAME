#! /usr/bin/env python

from PyQt4.QtGui import *
from PyQt4.QtCore import *
from ImageSource import *
from RoiEditor import *
import cv2
import os.path
from color import *
from Plugin import *
from FileWindow import *
import Settings
import math

class GuiWindow(QMainWindow):
    # state for self.playStat
    V_PAUSE = 0
    V_PLAYING = 1
    
    def __init__(self,videofile='',jsonfile=''):
        QMainWindow.__init__(self)
        self.settings = Settings.loadSettings();

        # play state
        self.playStat = GuiWindow.V_PAUSE

        self.videofile = videofile
        self.jsonfile = jsonfile
        self.txtfile = ''
        self.imgSource = ImageSource(self.videofile,self.jsonfile)
        self.plugin = Plugin(self.imgSource)
        self.lastPluginFnum = -1
        
        self.frameW = self.settings['frameW']
        self.frameH = self.settings['frameH'] # will be updated when open video file
        self.scale = 1.0
        self.offset = [0,0]
        self.zoomed = False

        self.holdShift = False
        self.holdControl = False
        self.holdAlt = False
        
        self.labelfile = ''
        #
        self.rate = 30
        self.fnum = 0
        self.focusRoiId = -1
        self.selRoiId = -1
        self.fcnt = self.imgSource.getFrameCount()
        self.roiLabels=(
            # name, editable
            ("label_name",False),
            ("label_id",  False),
            ("roi_score", False),
            ("roi_x",     True),
            ("roi_y",     True),
            ("roi_w",     True), 
            ("roi_h",     True), 
            ("roi_id",    False)
        )
        
        # setup timer tick
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000/self.rate)
        self.liftPressed = False
        # init gui
        self.guiInit()
        self.updateImage(self.fnum,True)

    def guiInit(self):
        # create main menu
        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        # File
        fileMenu = mainMenu.addMenu('&File')
        # add Open to File
        openButton = QAction('Open...',self)
        openButton.triggered.connect(self.openFileWin)
        fileMenu.addAction(openButton)
        saveVideoButton = QAction('Save Video With ROIs',self)
        saveVideoButton.triggered.connect(self.saveVideo)
        fileMenu.addAction(saveVideoButton)
        saveJsonButton = QAction('Save Json',self)
        saveJsonButton.triggered.connect(self.saveJson)
        fileMenu.addAction(saveJsonButton)
        # add SaveMask to File
        saveMaskButton = QAction('Save Mask',self)
        saveMaskButton.triggered.connect(self.saveMask)
        fileMenu.addAction(saveMaskButton)
        # add exit to File
        exitButton = QAction('Exit',self)
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # settings
        settingsMenu = mainMenu.addMenu('&Settings')

        colorButton = QAction('Color',self)
        colorButton.triggered.connect(self.pickColor)
        settingsMenu.addAction(colorButton)

        bgColorButton = QAction('BgColor',self)
        bgColorButton.triggered.connect(self.pickBgColor)
        settingsMenu.addAction(bgColorButton)

        focusColorButton = QAction('FocusColor',self)
        focusColorButton.triggered.connect(self.pickFocusColor)
        settingsMenu.addAction(focusColorButton)

        focusBgColorButton = QAction('FocusBgColor',self)
        focusBgColorButton.triggered.connect(self.pickFocusBgColor)
        settingsMenu.addAction(focusBgColorButton)

        selColorButton = QAction('SelColor',self)
        selColorButton.triggered.connect(self.pickSelColor)
        settingsMenu.addAction(selColorButton)

        saveSettingsButton = QAction('Save',self)
        saveSettingsButton.triggered.connect(self.saveSettings)
        settingsMenu.addAction(saveSettingsButton)

        # help
        helpMenu = mainMenu.addMenu('&Help')

        # create layouts
        widget = QWidget(self)
        layout = QHBoxLayout(widget)

        widgetL = QWidget(self)
        layoutL = QVBoxLayout(widgetL)

        widgetR = QWidget(self)
        layoutR = QGridLayout(widgetR);
        layoutR.setAlignment(Qt.AlignTop)

        layout.addWidget(widgetL)
        layout.addWidget(widgetR)

        # left panel
        self.picture = QLabel(self)
        self.picture.setPixmap(QPixmap(self.frameW,self.frameH))

        self.slider = QSlider(Qt.Horizontal)
        self.slider.resize(self.frameW,10)
        self.slider.valueChanged.connect(self.sliderChanged)

        widgetCtrl = QWidget(self)
        layoutCtrl = QHBoxLayout(widgetCtrl);

        self.playButton = QPushButton('Play')
        self.playButton.clicked.connect(self.playClicked)
        self.nextButton = QPushButton('Next')
        self.nextButton.clicked.connect(self.nextClicked)
        self.lastButton = QPushButton('Last')
        self.lastButton.clicked.connect(self.lastClicked)
        self.beginButton = QPushButton('Begin')
        self.beginButton.clicked.connect(self.beginClicked)
        self.endButton = QPushButton('End')
        self.endButton.clicked.connect(self.endClicked)
        self.gotoLabel = QLabel(self)
        self.gotoLabel.setText('  fnum:')
        self.gotoInput = QLineEdit(self)
        self.gotoInput.setText('0')
        self.gotoInput.editingFinished.connect(self.gotoChanged)
        layoutCtrl.addWidget(self.beginButton)
        layoutCtrl.addWidget(self.lastButton)
        layoutCtrl.addWidget(self.playButton)
        layoutCtrl.addWidget(self.nextButton)
        layoutCtrl.addWidget(self.endButton)
        layoutCtrl.addWidget(self.gotoLabel)
        layoutCtrl.addWidget(self.gotoInput)

        layoutL.addWidget(self.picture)
        layoutL.addWidget(self.slider)
        layoutL.addWidget(widgetCtrl)

        # right panel
        space = QLabel(self)
        layoutR.addWidget(space,0,0)
        
        self.RoiInfo = {}
        for i in range(8):
            label = QLabel(self)
            label.setText(self.roiLabels[i][0])
            if self.roiLabels[i][1]:
                edit = QLineEdit(self)
                edit.editingFinished.connect(self.inputChanged)
            elif self.roiLabels[i][0] == 'label_name':
                edit = QPushButton(self)
                edit.clicked.connect(self.chLabelClicked)
            else:
                edit = QLabel(self)
            edit.setFixedWidth(150)
            self.RoiInfo[self.roiLabels[i][0]] = edit
            layoutR.addWidget(label,i+1,0)
            layoutR.addWidget(edit,i+1,1)

        space = QLabel(self)
        layoutR.addWidget(space,9,0)
        
        self.addButton = QPushButton('Add ROI')
        self.addButton.setCheckable(True)
        self.addButton.clicked.connect(self.addClicked)
        layoutR.addWidget(self.addButton,10,0)
        self.delButton = QPushButton('Delete ROI')
        self.delButton.clicked.connect(self.delClicked)
        layoutR.addWidget(self.delButton,11,0)

        self.tmplMatchButton = QPushButton('TmplMatch')
        self.tmplMatchButton.setCheckable(True)
        self.tmplMatchButton.clicked.connect(self.tmplMatch)
        layoutR.addWidget(self.tmplMatchButton,12,0)
        
        self.slider.setValue(0)
        self.setCentralWidget(widget)        

    def openFileWin(self):
        self.fileWin = FileWindow(self)
        self.fileWin.show()
        
    def openVideo(self):
        self.videofile = str(QFileDialog.getOpenFileName(self))
        self.updateVideo(self.videofile)
        
    def updateVideo(self,filename):
        self.imgSource.openVideo(filename)
        self.videoParamUpdate()
        
    def openImgDir(self):
        dirname = str(QFileDialog.getExistingDirectory(self))
        self.updateImgDir(dirname)

    def updateImgDir(self,dirname):
        self.imgSource.openImgDir(dirname)
        self.videoParamUpdate()

    def videoParamUpdate(self):
        self.fcnt = self.imgSource.getFrameCount()
        self.fnum = 0
        self.slider.setValue(self.fnum) # will call sliderChanged
        (w,h) = self.imgSource.getSize()
        self.imgSize = (w,h)
        self.scale = float(self.frameW) / w
        self.frameW = int(w*self.scale+0.5)
        self.frameH = int(h*self.scale+0.5)
        self.updateImage(self.fnum,True)
        
    def openJson(self):
        self.jsonfile = str(QFileDialog.getOpenFileName(self))
        self.updateJson(self.jsonfile)
        
    def updateJson(self,filename):
        self.imgSource.openRoiJson(filename)
        self.lastPluginFnum = -1
        self.updateImage(self.fnum,True)

    def updateTxt(self,filename):
        self.imgSource.openRoiTxt(filename)
        self.lastPluginFnum = -1
        self.updateImage(self.fnum,True)
        
    def openLabel(self):
        self.labelfile = str(QFileDialog.getOpenFileName(self))
        self.updateLabel(self.labelfile)
        
    def updateLabel(self,filename):
        if not os.path.isfile(filename):
            return
        self.imgSource.openLabel(filename)
        self.imgSource.updateRoiLabel()

    def openUpdateDir(self):
        dirname = str(QFileDialog.getExistingDirectory(self))
        self.imgSource.updateInfoFromDir(dirname)
        
    def saveVideo(self):
        self.imgSource.saveVideo()

    def saveJson(self):
        self.imgSource.saveJson()

    def saveMask(self):
        self.imgSource.saveMask()

    def pickColor(self):
        color = QColorDialog.getColor()
        self.settings['Color'] = (color.red(),color.green(),color.blue())
        self.updateImage(self.fnum,False)

    def pickBgColor(self):
        color = QColorDialog.getColor()
        self.settings['BgColor'] = (color.red(),color.green(),color.blue())
        self.updateImage(self.fnum,False)

    def pickFocusColor(self):
        color = QColorDialog.getColor()
        self.settings['FocusColor'] = (color.red(),color.green(),color.blue())
        self.updateImage(self.fnum,False)

    def pickFocusBgColor(self):
        color = QColorDialog.getColor()
        self.settings['FocusBgColor'] = (color.red(),color.green(),color.blue())
        self.updateImage(self.fnum,False)

    def pickSelColor(self):
        color = QColorDialog.getColor()
        self.settings['SelColor'] = (color.red(),color.green(),color.blue())
        self.updateImage(self.fnum,False)

    def saveSettings(self):
        Settings.saveSettings(self.settings)

    def distance(self,pos1,pos2):
        dx = pos1[0]-pos2[0]
        dy = pos1[1]-pos2[1]
        return math.sqrt(dx*dx+dy*dy)
    
    def eventFilter(self, source, event):
        """ pick up mouse event """
        if self.playStat == GuiWindow.V_PAUSE and self.imgSource.roisInputEnable and source == self.picture:
            update = False
            if event.type() == QEvent.MouseMove:
                pos = event.pos()
                (id,touch) = self.roiEditor.touched(
                    (int((pos.x()-self.offset[0])/self.scale),
                     int((pos.y()-self.offset[1])/self.scale)))
                self.updateCursor(id,touch)
                if event.buttons() == Qt.LeftButton:
                    update,self.selRoiId = self.roiEditor.dragging(
                        (int((pos.x()-self.offset[0])/self.scale),
                         int((pos.y()-self.offset[1])/self.scale)),
                        self.selRoiId)
                elif self.roiEditor.dragged:
                    update,self.selRoiId = self.roiEditor.dragging((),self.selRoiId)
                if id != self.focusRoiId:
                    self.focusRoiId = id
                    update = True
            elif event.type() == QEvent.MouseButtonPress:
                if event.buttons() == Qt.LeftButton:
                    source.setFocus()
                    self.liftPressed = True
                    self.pressedPos = event.pos()
            elif event.type() == QEvent.MouseButtonRelease:
                pos = event.pos()
                if self.liftPressed:
                    distance = self.distance((pos.x(),pos.y()),
                                             (self.pressedPos.x(),
                                              self.pressedPos.y()))
                    # select only press and release at same position
                    if distance < 10:
                        # pos on real image
                        transPos = (int((pos.x()-self.offset[0])/self.scale),
                                    int((pos.y()-self.offset[1])/self.scale))

                        if self.roiEditor.addVertexState:
                            # add vertex only if mouse not in movinge
                            self.roiEditor.addVertex(transPos,self.selRoiId)
                            update = True
                        else:
                            (id,touch) = self.roiEditor.touched(transPos)
                            if id != self.selRoiId:
                                self.selRoiId = id
                                update = True
                    else:
                        update,self.selRoiId = self.roiEditor.dragging((),self.selRoiId)

                    if self.roiEditor.zoomState:
                        self.roiEditor.setZoomStateOff()
                        if distance > 10: # minimum zoom size
                            (ret,scale,offset) = self.zoom(
                                (self.roiEditor.zoomRoi[0],
                                 self.roiEditor.zoomRoi[1],
                                 self.roiEditor.zoomRoi[2]-self.roiEditor.zoomRoi[0],
                                 self.roiEditor.zoomRoi[3]-self.roiEditor.zoomRoi[1])
                            )
                            if ret:
                                self.zoomed = True
                                self.scale = scale
                                self.offset = offset
                        update = True

            if update:
                self.updateImage(self.fnum,False)
        return QMainWindow.eventFilter(self, source, event)

    def keyPressEvent(self,event):
        if self.holdShift:
            v = -1
        else:
            v = 1

        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.holdShift:
                self.delRois()
            else:
                self.delClicked()
        elif event.key() == Qt.Key_Up:
            pass
            #if self.holdControl:
            #    self.roiEditor.changeY(-1,self.focusRoiId)
            #else:
            #    self.roiEditor.changeY(-v,self.focusRoiId)
            #    self.roiEditor.changeH(v,self.focusRoiId)
        elif event.key() == Qt.Key_Down:
            pass
            #if self.holdControl:
            #    self.roiEditor.changeY(1,self.focusRoiId)
            #else:
            #    self.roiEditor.changeH(v,self.focusRoiId)
        elif event.key() == Qt.Key_Left:
            pass
            #if self.holdControl:
            #    self.roiEditor.changeX(-1,self.focusRoiId)
            #else:
            #    self.roiEditor.changeX(-v,self.focusRoiId)
            #    self.roiEditor.changeW(v,self.focusRoiId)
        elif event.key() == Qt.Key_Right:
            pass
            #if self.holdControl:
            #    self.roiEditor.changeX(1,self.focusRoiId)
            #else:
            #    self.roiEditor.changeW(v,self.focusRoiId)
        elif event.key() == Qt.Key_Space:
            if self.holdShift:
                self.lastClicked()
            else:
                self.nextClicked()
        elif event.key() == Qt.Key_Shift:
            self.holdShift = True
        elif event.key() == Qt.Key_Control:
            self.holdControl = True
            self.roiEditor.setAddVertexState(True,self.selRoiId)
        elif event.key() == Qt.Key_Alt:
            self.holdAlt = True
        elif event.key() == Qt.Key_Escape:
            self.zoomed = False
            (w,h) = self.imgSource.getSize()
            self.scale = float(self.frameW) / w
            self.offset = [0,0]
        self.updateImage(self.fnum,False)

            
    def keyReleaseEvent(self,event):
        if event.key() == Qt.Key_Alt:
            self.holdAlt = False
        elif event.key() == Qt.Key_Shift:
            self.holdShift = False
        elif event.key() == Qt.Key_Control:
            self.holdControl = False
            self.roiEditor.setAddVertexState(False,self.selRoiId)
        self.updateImage(self.fnum,False)

    def updateCursor(self,id,touch):
        if self.roiEditor.addVertexState and self.selRoiId >= 0:
            QApplication.setOverrideCursor(Qt.CrossCursor)
            return
        if id >= 0 and id == self.selRoiId:
            if touch == RoiEditor.TopLeft or touch == RoiEditor.BottomRight:
                QApplication.setOverrideCursor(Qt.SizeFDiagCursor)
            elif touch == RoiEditor.Top or touch == RoiEditor.Bottom:
                QApplication.setOverrideCursor(Qt.SizeVerCursor)
            elif touch == RoiEditor.TopRight or touch == RoiEditor.BottomLeft:
                QApplication.setOverrideCursor(Qt.SizeBDiagCursor)
            elif touch == RoiEditor.Left or touch == RoiEditor.Right:
                QApplication.setOverrideCursor(Qt.SizeHorCursor)
            elif touch == RoiEditor.Center:
                QApplication.setOverrideCursor(Qt.SizeAllCursor)
            else:
                QApplication.setOverrideCursor(Qt.ArrowCursor)
        else:
            QApplication.setOverrideCursor(Qt.ArrowCursor)

    def tick(self):
        """ timer callback to play """
        # if sources are opened
        self.slider.setValue(self.fnum) # will call sliderChanged
        if(self.playStat == GuiWindow.V_PLAYING):
            if self.fnum < self.fcnt-1:
                self.fnum += 1

    def playClicked(self):
        if self.playStat == GuiWindow.V_PLAYING:
            self.playStat = GuiWindow.V_PAUSE
            self.playButton.setText('Play')
        else:
            self.playStat = GuiWindow.V_PLAYING
            self.playButton.setText('Pause')

    def lastClicked(self):
        if self.fnum>0:
            self.fnum -= 1
            self.slider.setValue(self.fnum) # will call sliderChanged

    def nextClicked(self):
        if self.fnum < self.fcnt-1:
            self.fnum += 1
            self.slider.setValue(self.fnum) # will call sliderChanged

    def beginClicked(self):
        self.fnum = 0
        self.slider.setValue(self.fnum) # will call sliderChanged

    def endClicked(self):
        self.fnum = self.fcnt - 1
        self.slider.setValue(self.fnum) # will call sliderChanged

    def delClicked(self):
        if self.imgSource.roisInputEnable:
            self.roiEditor.delRoi(self.selRoiId)
            self.updateImage(self.fnum,False)

    def delRois(self):
        """ delete all rois in view except selected """
        if self.imgSource.roisInputEnable:
            self.roiEditor.delRois(self.zoomed,self.selRoiId)
            self.updateImage(self.fnum,False)
            


    def addClicked(self):
        if self.imgSource.roisInputEnable:
            self.roiEditor.setAddRoiState(self.addButton.isChecked())
            self.selRoiId = -1
            self.updateImage(self.fnum,False)

    def tmplMatch(self):
        self.plugin.setTmplMatchEnable(self.tmplMatchButton.isChecked())
    
    def gotoChanged(self):
        v = int(self.gotoInput.text())
        if v < 0:
            self.fnum = 0
        elif v >= self.fcnt:
            self.fnum = self.fcnt
        else:
            self.fnum = v
        self.slider.setValue(self.fnum) # will call sliderChanged

    def inputChanged(self):
        if self.imgSource.roisInputEnable:
            self.roiEditor.setValues(self.RoiInfo,self.selRoiId)
            self.updateImage(self.fnum,False)
        
    def chLabelClicked(self):
        if self.selRoiId <0:
            return
        items = self.imgSource.getLabelNameList()
        qitem,ok = QInputDialog.getItem(self,'Label_Name selection','list of labels',items,0,False)
        if ok:
            item = str(qitem)
            ret,id = self.imgSource.getLabelId(item)
            self.RoiInfo['label_name'].setText(item)
            self.RoiInfo['label_id'].setText(str(id))
            self.inputChanged()
        
        
    def sliderChanged(self):
        # if sources are opened
        self.fnum = self.slider.value()
        self.selRoiId = -1
        self.gotoInput.setText(str(self.fnum))
        self.updateImage(self.fnum,True)

    def zoom(self,roi):
        ''' roi: (x,y,w,h) return (ret,scale,offset) '''
        if roi[2] == 0 or roi[3] == 0:
            return (False,1.0,(0,0))
        x,y,w,h = float(roi[0]),float(roi[1]),float(roi[2]),float(roi[3])
        r = float(self.frameW)/float(self.frameH)
        if w/h >= r:
            w1,h1 = w,w/r
            x1,y1 = x,y-(h1-h)/2
        else:
            w1,h1 = h*r,h 
            x1,y1 = x-(w1-w)/2,y
        scale = float(self.frameW) / w1
        return (True,scale,(-x1*scale,-y1*scale))
        

    def updateImage(self,fnum,newEditor):
        if self.lastPluginFnum == fnum:
            ret1,img = self.imgSource.getImage(fnum)
            ret2,rois = self.imgSource.getRois(fnum)
        else:
            (ret1,img),(ret2,rois) = self.plugin.beginProcess(fnum)
            if ret1 and ret2:
                self.lastPluginFnum = fnum
            newEditor = True
            
        if ret2 and newEditor:
            self.roiEditor = RoiEditor(rois,self.imgSize)
            self.addButton.setChecked(False)
        if ret1:
            w,h = self.frameW,self.frameH
            affine = np.float32([[self.scale,0,self.offset[0]],
                                 [0,self.scale,self.offset[1]]])
            img1 = cv2.warpAffine(img,affine,(w,h))
            img2 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
            if ret2:
                self.drawRois(img2,rois,self.selRoiId,self.focusRoiId)
            img3 = QImage(img2,w,h,QImage.Format_RGB888)
            self.picture.setPixmap(QPixmap.fromImage(img3))
            self.slider.resize(w,10)
            self.slider.setMaximum(self.fcnt-1)
            self.addButton.setChecked(self.roiEditor.addRoiState)

                    
    def drawRois(self,frame,rois,selRoiId,focusId):
        edge = 3
        hasSel = False
        for roi in rois:
            x0,y0 = self.posToScr((roi['roi_x'],roi['roi_y']))
            w,h  = self.sizeToScr((roi['roi_w'],roi['roi_h']))
            x1 = x0 + w - 1
            y1 = y0 + h - 1
            label_id = roi['roi_label']['label_id']
            label_name = roi['roi_label']['label_name']

            if roi['roi_id'] == selRoiId:
                hasSel = True
                self.updateRoiInfo(roi)
                self.drawSelRoi(frame,roi,edge)
            elif roi['roi_id'] == focusId:
                if label_id > 0:
                    cv2.rectangle(frame,(x0,y0),(x1,y1),
                                  self.settings['FocusColor'],1)
                else:
                    cv2.rectangle(frame,(x0,y0),(x1,y1),
                                  self.settings['FocusBgColor'],1)
            else:
                if label_id > 0:
                    cv2.rectangle(frame,(x0,y0),(x1,y1),
                                  self.settings['Color'],1)
                    #cv2.rectangle(frame,(x0,y0),(x1,y1),
                    #              ColorTable[getColor(label_id)],1)
                else:
                    cv2.rectangle(frame,(x0,y0),(x1,y1),self.
                                  settings['BgColor'],1)

            self.drawVertex(frame,roi)
            # end of for

        if not hasSel:
            self.clearRoiInfo()
            
        if self.imgSource.roisInputEnable and self.roiEditor.zoomState:
            cv2.rectangle(frame,
                          (int(self.roiEditor.zoomRoi[0]*self.scale+self.offset[0]),
                           int(self.roiEditor.zoomRoi[1]*self.scale+self.offset[1])),
                          (int(self.roiEditor.zoomRoi[2]*self.scale+self.offset[0]),
                           int(self.roiEditor.zoomRoi[3]*self.scale+self.offset[1])),
                          (200,0,200),1)

    def drawSelRoi(self,frame,roi,edge=3):
        x0,y0 = self.posToScr((roi['roi_x'],roi['roi_y']))
        w,h  = self.sizeToScr((roi['roi_w'],roi['roi_h']))
        x1 = x0 + w - 1
        y1 = y0 + h - 1
        cv2.rectangle(frame,(x0,y0),(x1,y1),self.settings['SelColor'],1)
        if not self.holdControl:
            cv2.circle(frame,(x0,y0),edge+1,self.settings['SelColor'],1)
            cv2.circle(frame,(x0,y1),edge+1,self.settings['SelColor'],1)
            cv2.circle(frame,(x1,y0),edge+1,self.settings['SelColor'],1)
            cv2.circle(frame,(x1,y1),edge+1,self.settings['SelColor'],1)
            cv2.rectangle(frame,
                          (x0+w/2-edge,y0-edge),
                          (x0+w/2+edge,y0+edge),
                          self.settings['SelColor'],1)
            cv2.rectangle(frame,
                          (x0-edge,y0+h/2-edge),
                          (x0+edge,y0+h/2+edge),
                          self.settings['SelColor'],1)
            cv2.rectangle(frame,
                          (x1-edge,y0+h/2-edge),
                          (x1+edge,y0+h/2+edge),
                          self.settings['SelColor'],1)
            cv2.rectangle(frame,
                          (x0+w/2-edge,y1-edge),
                          (x0+edge+w/2,y1+edge),
                          self.settings['SelColor'],1)

    def drawVertex(self,frame,roi,edge=3):
        if not 'vertices' in roi:
            return
        beginPos = ()
        lastPos = ()
        for pos in roi['vertices']:
            scrPos = self.posToScr(pos)
            if self.roiEditor.addVertexState:
                cv2.circle(frame,scrPos,edge+1,self.settings['VertexColor'],1)
            if len(beginPos) == 0:
                beginPos = scrPos
            if len(lastPos) > 0:
                cv2.line(frame,lastPos,scrPos,self.settings['VertexColor'],1)
            lastPos = scrPos
        if len(beginPos)> 0 and len(lastPos)> 0:
            cv2.line(frame,lastPos,beginPos,self.settings['VertexColor'],1)
     
    def posToScr(self,pos):
        """ convert pos to screen pos """
        return (int(pos[0]*self.scale+self.offset[0]),
                int(pos[1]*self.scale+self.offset[1]))

    def sizeToScr(self,size):
        """ convert size to screen size """
        return (int(size[0]*self.scale),
                int(size[1]*self.scale))
                
    def clearRoiInfo(self):
        for key,_ in self.roiLabels:
            self.RoiInfo[key].setText('')

    def updateRoiInfo(self,roi):
        for key,value in roi.iteritems():
            if key == "roi_label":
                for key1,value1 in value.iteritems():
                    self.RoiInfo[key1].setText(str(value1))
            elif key == "vertices":
                pass #todo
            else:
                self.RoiInfo[key].setText(str(value))
                    
            

