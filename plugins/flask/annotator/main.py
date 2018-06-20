#! /usr/bin/env python
from PyQt4.QtGui import *
from GuiWindow import *
import os

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    videofile = ''
    jsonfile = ''
    for file in sys.argv:
        filename,ext = os.path.splitext(file)
        if ext == '.avi' and videofile == '':
            videofile = file
        if ext == '.json' and jsonfile == '':
            jsonfile = file
    win = GuiWindow(videofile,jsonfile)
    win.show()
    app.installEventFilter(win)
    sys.exit(app.exec_())
