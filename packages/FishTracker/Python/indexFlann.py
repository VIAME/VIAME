import cv2
from pyflann import *
from numpy import *
from numpy.random import *
import numpy as np

#http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann_manual-1.8.4.pdf
#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
class IndexFlann():
    def __init__(self, page_max, labels, threshold):
        self.labels = labels;
        self.k = page_max
        self.threshold = max(min(1.0, threshold), 0.0)
        self.reset()
        
    def reset(self):
        self.dataset = [];
        self.pageset = np.array([]);
        self.page = long(0)
        self.count = long(0)
        self.dic = {}
        
    def addFeatures(self, feature):
        fs = len(feature)
        self.page = self.count / self.k
        if (self.count % self.k) == 0:
            self.pageset = np.array([]);
            self.pageset = np.array(feature)
        else :
            self.pageset = np.vstack([self.pageset, np.array(feature)]);
        if (self.count % self.k) == self.k -1:
                self.dataset += [self.pageset]; 
                
        self.count = self.count+1;
        
    def trimCandicates_byValue(self, score_max):
        if len(self.dic) == 0:
            return 0
            
        thresh = score_max * self.threshold
        
        for k, v in self.dic.items():
            some_dict = [(key, value) for key, value in v if value < thresh]
            #{key: value for key, value in v if value < thresh}
            self.dic[k] = some_dict
        return self.dic;
    
    def trimCandicates_byNumber(self):
        if len(self.dic) == 0:
            return 0
        
        for k, v in self.dic.items():
            some_dict = v[0:min(self.labels, len(v))]
            self.dic[k] = some_dict
        return self.dic;  
      
    def nearestneighbor(self, ntype='byNumber'):   
        flann = FLANN();
        algo_type = 'kdtree' # 'kmeans'
        index = {};
        score = {};        
        if self.count == 0:
            return 0, 0
        if self.page == len(self.dataset):
            self.dataset += [self.pageset]; 
                  
        for i in range(self.page+1):    
            indices = []
            scores = []
            for j in range(self.page+1): 
                result, dists = flann.nn(self.dataset[j], self.dataset[i], 
                        self.labels, algorithm=algo_type, branching=32, iterations=7, checks=16)
                result = result + self.k * j;
                indices += [result]
                scores += [dists]
            indices = np.array(indices)
            scores = np.array(scores)
            index[i] = hstack(indices)
            score[i] = hstack(scores);
            
        score_max = np.max([np.max(score[i]) for i in range(self.page+1)])
       
        for i in range(self.page+1):
            for j in range(len(self.dataset[i])):
                m = i * self.k + j
                temp = dict(zip(index[i][j], score[i][j]))
                self.dic[m] = sorted(temp.items(), key=lambda x: x[1])
        if ntype == 'byNumber':
            self.dic = self.trimCandicates_byNumber()
        elif ntype == 'byValue':
            self.dic = self.trimCandicates_byValue(score_max);
        
        return self.dic;    
            
    def nearestneighbor_cv(self, ntype='byNumber'): 
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 32)
        search_params = dict(checks=128)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)  
          
        index = {};
        score = {};        
        if self.count == 0:
            return 0, 0
        if self.page == len(self.dataset):
            self.dataset += [self.pageset]; 
            
        labels_num = np.minimum(len(self.dataset[0]), self.labels) 
        print 'label numbers =  ', labels_num   
 
                  
        for i in range(self.page+1):    
            indices = []
            scores = []
            for j in range(self.page+1): 
                data_ref = np.float32(self.dataset[i])
                data_ins = np.float32(self.dataset[j])                
                matches = flann.knnMatch(data_ref, data_ins,labels_num) # ref: query, ins: train
                result = [[] for m in range(len(matches))]
                dists = [[] for m in range(len(matches))]
                # Need to draw only good matches, so create a mask
                for m in range(len(matches)):
                    matches[m] = sorted(matches[m], key = lambda x:x.distance)
                    result[m] = [matches[m][n].trainIdx for n in range(len(matches[m]))]
                    dists[m] =  [matches[m][n].distance for n in range(len(matches[m]))]
                result = np.array(result) + self.k * j;
                indices += [result]
                scores += [dists]
            indices = np.array(indices)
            scores = np.array(scores)
            index[i] = hstack(indices)
            score[i] = hstack(scores);   
            
        score_max = np.max([np.max(score[i]) for i in range(self.page+1)])
        
        for i in range(self.page+1):
            for j in range(len(self.dataset[i])):
                m = i * self.k + j
                temp = dict(zip(index[i][j], score[i][j]))
                self.dic[m] = sorted(temp.items(), key=lambda x: x[1])
        if ntype == 'byNumber':
            self.dic = self.trimCandicates_byNumber()
        elif ntype == 'byValue':
            self.dic = self.trimCandicates_byValue(score_max);            
        return self.dic;    


              