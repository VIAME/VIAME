#!/usr/bin/python
import os,sys,json,shutil,fileinput
import numpy as np
import ImageIO

class CaffeTrain:
    def __init__(self,config,wrkDir=None):
        self.curDir = os.path.dirname(__file__)
        if wrkDir == None:
            wrkDir  = config["workDir"]
        self.wrkDir = os.path.abspath(wrkDir)
        self.config  = config["CaffeTrain"]
        self.caffeModelDir = self.wrkDir + '/CaffeModel'
        ImageIO.mkdir_p(self.caffeModelDir)

        self.trainNetFile = self.caffeModelDir + '/' + self.config['trainNetFile']
        self.meanFile = self.caffeModelDir + '/' + self.config['meanFile']
        self.modelFile = self.caffeModelDir + '/' + self.config['modelFile']
        self.solverFile = self.caffeModelDir+'/solver.prototxt'
        self.caffe_exe = self.curDir+'/../local/bin/caffe'

    def run():
        os.system("%s train -solver %s -weights %s -gpu 0" % \
                  (self.caffe_exe,self.solverFile,self.modelFile))
        

    def createSolverFile(self):
        """ runtime create solver file based on configuration"""
        f = open(self.solverFile,"w");
        f.write("net: "+ self.trainNetFile +'\n')
        for key,val in self.config["solver"].iteritems():
            f.write("%s : %s\n"%(key,val))
        f.write("snapshot_prefix: " + self.wrkDir + '/' + 'snapshots\n')
        f.close()

    def updateModelFile(self):
        """ runtime update model file """
        configDir = os.path.dirname(__file__) + '/../bin/config'
        if not os.path.isfile(self.meanFile):
            shutil.copyfile(configDir+'/imagenet_mean.binaryproto',self.meanFile)

        if not os.path.isfile(self.modelFile):
            shutil.copyfile(configDir+'/bvlc_reference_caffenet.caffemodel',self.modelFile)

        if not os.path.isfile(self.trainNetFile):
            shutil.copyfile(configDir+'/train_val0.prototxt',self.trainNetFile)
            
        # update mean_file and source/test in trainNetFile
        with open(self.trainNetFile,"r") as fd:
            lines = fd.readlines()
        with open(self.trainNetFile,"w") as fd:
            train = True
            for line in lines:
                toks = line.split()
                if  toks[0] == 'mean_file:': 
                    fd.write("    %s %s\n"%(toks[0],self.meanFile))
                elif toks[0] == 'source:':
                    if train:
                        fd.write("    %s %s\n"%(toks[0],self.caffeModelDir+'/train.txt'))
                        train = False
                    else:
                        fd.write("    %s %s\n"%(toks[0],self.caffeModelDir+'/test.txt'))
                else:
                    fd.write(line)


#######################################################
# module test                
def main(argv):
    curDir = os.path.dirname(__file__)
    configFile = curDir+'/../bin/config/sysConfig.json' 
    with open(configFile) as fd:
        config = json.load(fd)

    train = CaffeTrain(config)
    train.createSolverFile()
    train.updateModelFile()

if __name__ == "__main__":
    main(sys.argv)



