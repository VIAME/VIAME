#! /usr/bin/env python
import os,json

def loadSettings(filename='settings.json'):
    if os.path.isfile(filename):
        with open(filename) as fd:
            settings = json.load(fd)
    else:
        settings = loadDefaultSettings()
        saveSettings(settings,filename)
    return settings

def loadDefaultSettings():
    return {
        # image frame size
        'frameW':640,
        'frameH':480,
        # ROI colors
        'SelColor':(0,255,0),     # selected ROI
        'FocusColor':(255,255,0), # when mouse move to a ROI
        'FocusBgColor':(255,0,0), # when mouse move to a ROI, not classified
        'Color':(200,200,0),      # classified ROI
        'BgColor':(200,0,0),      # background or not classified
        'VertexColor':(0,255,0)   # vertex color
    }

def saveSettings(settings,filename='settings.json'):
    with open(filename,'w') as fp:
        json.dump(settings,fp,sort_keys=True,indent=4, separators=(',', ': '))


