#! /usr/bin/env python
from PyQt4.QtGui import *
from PyQt4.QtCore import *
from ImageSource import *

class FileWindow(QMainWindow):
    def __init__(self, parent):
        QMainWindow.__init__(self)
        self.parent = parent
        self.videoname = ''
        self.imagename = ''
        self.jsonname = ''
        self.txtname = ''
        self.labelname = ''
        self.udirname = ''
        self.guiInit()

    def guiInit(self):
        widget = QWidget(self)
        layout = QGridLayout(widget)
        layout.setAlignment(Qt.AlignTop)

        line = 0
        inWidth = 300
        #space = QLabel(self)
        #layout.addWidget(space,line,0)
        #line += 1

        # video file
        vLabel = QLabel(self)
        vLabel.setText(' Video File:')
        layout.addWidget(vLabel,line,0)
        self.videoInput = QLineEdit(self)
        self.videoInput.editingFinished.connect(self.videoInputChanged)
        self.videoInput.setFixedWidth(inWidth)
        layout.addWidget(self.videoInput,line,1)
        videoButton = QPushButton('Browse')
        videoButton.clicked.connect(self.videoInputBrowse)
        layout.addWidget(videoButton,line,2)
        line += 1

        # or image directory
        iLabel = QLabel(self)
        iLabel.setText(' or Images:')
        layout.addWidget(iLabel,line,0)
        self.imageInput = QLineEdit(self)
        self.imageInput.editingFinished.connect(self.imageInputChanged)
        self.imageInput.setFixedWidth(inWidth)
        layout.addWidget(self.imageInput,line,1)
        imageButton = QPushButton('Browse')
        imageButton.clicked.connect(self.imageInputBrowse)
        layout.addWidget(imageButton,line,2)
        line += 1

        # roi json
        jLabel = QLabel(self)
        jLabel.setText(' ROI Json File:')
        layout.addWidget(jLabel,line,0)
        self.jsonInput = QLineEdit(self)
        self.jsonInput.editingFinished.connect(self.jsonInputChanged)
        self.jsonInput.setFixedWidth(inWidth)
        layout.addWidget(self.jsonInput,line,1)
        jsonButton = QPushButton('Browse')
        jsonButton.clicked.connect(self.jsonInputBrowse)
        layout.addWidget(jsonButton,line,2)
        line += 1

        # roi txt
        tLabel = QLabel(self)
        tLabel.setText('or ROI txt File:')
        layout.addWidget(tLabel,line,0)
        self.txtInput = QLineEdit(self)
        self.txtInput.editingFinished.connect(self.txtInputChanged)
        self.txtInput.setFixedWidth(inWidth)
        layout.addWidget(self.txtInput,line,1)
        txtButton = QPushButton('Browse')
        txtButton.clicked.connect(self.txtInputBrowse)
        layout.addWidget(txtButton,line,2)
        line += 1

        # label json
        lLabel = QLabel(self)
        lLabel.setText(' Label Json File:')
        layout.addWidget(lLabel,line,0)
        self.labelInput = QLineEdit(self)
        self.labelInput.editingFinished.connect(self.labelInputChanged)
        self.labelInput.setFixedWidth(inWidth)
        layout.addWidget(self.labelInput,line,1)
        labelButton = QPushButton('Browse')
        labelButton.clicked.connect(self.labelInputBrowse)
        layout.addWidget(labelButton,line,2)
        line += 1
        
        # update directory
        uLabel = QLabel(self)
        uLabel.setText('Update Directory:')
        layout.addWidget(uLabel,line,0)
        self.udirInput = QLineEdit(self)
        self.udirInput.editingFinished.connect(self.udirInputChanged)
        self.udirInput.setFixedWidth(inWidth)
        layout.addWidget(self.udirInput,line,1)
        udirButton = QPushButton('Browse')
        udirButton.clicked.connect(self.udirInputBrowse)
        layout.addWidget(udirButton,line,2)
        line += 1

        # ok
        okButton = QPushButton('OK')
        okButton.clicked.connect(self.accept)
        layout.addWidget(okButton,line,2)
        
        self.setCentralWidget(widget)        

    def videoInputChanged(self):
        self.videoname = str(self.videoInput.text())
        
    def videoInputBrowse(self):
        self.videoname = str(QFileDialog.getOpenFileName(self,'Select Video File (.avi)'))
        self.videoInput.setText(self.videoname)
        self.imagename = ''
        self.imageInput.setText(self.imagename)

    def imageInputChanged(self):
        self.imagename = str(self.imageInput.text())
        
    def imageInputBrowse(self):
        self.imagename = str(QFileDialog.getExistingDirectory(self,'Select Image Directory'))
        self.imageInput.setText(self.imagename)
        self.videoname = ''
        self.videoInput.setText(self.videoname)
        
    def jsonInputChanged(self):
        self.jsonname = str(self.jsonInput.text())
        
    def jsonInputBrowse(self):
        if len(self.imagename) > 0:
            dirname = os.path.dirname(self.imagename)
        elif len(self.videoname) > 0:
            dirname = os.path.dirname(self.videoname)
        else:
            dirname = '.'
            
        self.jsonname = str(QFileDialog.getOpenFileName(self,
                                                        'Select ROI Json File',
                                                        dirname,
                                                        'images (*.json)'))
        self.jsonInput.setText(self.jsonname)
        self.txtInput.setText('')

    def txtInputChanged(self):
        self.txtname = str(self.txtInput.text())
        
    def txtInputBrowse(self):
        if len(self.imagename) > 0:
            dirname = os.path.dirname(self.imagename)
        elif len(self.videoname) > 0:
            dirname = os.path.dirname(self.videoname)
        else:
            dirname = '.'
            
        self.txtname = str(QFileDialog.getOpenFileName(self,
                                                       'Select ROI txt File',
                                                       dirname,
                                                       'images (*.txt)'))
        self.txtInput.setText(self.txtname)
        self.jsonInput.setText('')
        
    def labelInputChanged(self):
        self.labelname = str(self.labelInput.text())
        
    def labelInputBrowse(self):
        if len(self.imagename) > 0:
            dirname = os.path.dirname(self.imagename)
        elif len(self.videoname) > 0:
            dirname = os.path.dirname(self.videoname)
        else:
            dirname = '.'
            
        self.labelname = str(QFileDialog.getOpenFileName(self,
                                                        'Select Label Json File',
                                                        dirname,
                                                        'images (*.json)'))
        self.labelInput.setText(self.labelname)
        
    def udirInputChanged(self):
        self.udirname = str(self.udirInput.text())
        
    def udirInputBrowse(self):
        self.udirname = str(QFileDialog.getExistingDirectory(self,'Select Directory'))
        self.udirInput.setText(self.udirname)

    def accept(self):
        hasVideo = False
        if len(self.videoname) > 0:
            #print('video',self.videoname)
            self.parent.videofile = self.videoname
            self.parent.updateVideo(self.parent.videofile)
            hasVideo = True
        elif len(self.imagename) > 0:
            #print('image',self.imagename)
            self.parent.updateImgDir(self.imagename)
        if hasVideo and (len(self.jsonname) > 0 or len(self.txtname) > 0):
            if len(self.jsonname) > 0:
                #print('json',self.jsonname)
                self.parent.jsonfile = str(self.jsonname)
                self.parent.updateJson(self.parent.jsonfile)
            if len(self.txtname) > 0:
                #print('txt',self.txtname)
                self.parent.txtfile = str(self.txtname)
                self.parent.updateTxt(self.parent.txtfile)
            if len(self.labelname)>0:
                #print('label',self.labelname)
                self.parent.labelfile = self.labelname
                self.parent.updateLabel(self.parent.labelfile)
            if len(self.udirname)>0:
                #print('udir',self.udirname)
                self.parent.imgSource.updateInfoFromDir(self.udirname)
            self.close()

