# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:39:31 2017

@author: David Zhang
"""


from sklearn.cluster import KMeans
import scipy.sparse as sps
import scipy.spatial.distance as spd
from numpy import *
from numpy.random import *
import numpy as np
#import matplotlib.pyplot as plt
#import kmedoids
from scipy.sparse.linalg import eigsh
import scipy
import xml.etree.ElementTree
import os, time
import json
from math import*
from decimal import Decimal
from indexFlann import IndexFlann
import cv2
import io

#http://dataaspirant.com/2015/04/11/five-most-popular-similarity-measures-implementation-in-python/
class Similarity():
    def euclidean_distance(self, x,y):
       return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))
    def manhattan_distance(self,x,y):
       return sum(abs(a-b) for a,b in zip(x,y))
    def nth_root(self,value, n_root):
       root_value = 1/float(n_root)
       return round (Decimal(value) ** Decimal(root_value),3)
     
    def minkowski_distance(self,x,y,p_value):
       return self.nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)
    
    def square_rooted(self,x):
       return round(sqrt(sum([a*a for a in x])),3)
     
    def cosine_similarity(self,x,y):
       numerator = sum(a*b for a,b in zip(x,y))
       denominator = self.square_rooted(x)* self.square_rooted(y)
       return round(numerator/float(denominator),3)
    
    def jaccard_similarity(self,x,y):
       intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
       union_cardinality = len(set.union(*[set(x), set(y)]))
       return intersection_cardinality/float(union_cardinality)

#####################
def rowNormalize(D):
   row_norm= np.linalg.norm(D,axis=1);
   for r in range(D.shape[0]):
       D[r,:] = D[r,:]/row_norm[r]
   
   return D
## knee computation as described in the paper: "Finding a “Kneedle” in a Haystack: Detecting Knee Points in System Behavior"
def getClustersBasedonKnee(lam):
   x = np.asarray([i for i in range(np.prod(lam.shape))], dtype=float)
   xmax = np.max(x)
   if xmax != 0.0:
        x = x/(np.max(x))
   y = lam
   ymin = np.min(y)
   yrange = np.max(y) - ymin
   if yrange != 0.0:
        y = (y-ymin)/yrange
   delta =y-x
   knee = np.argmax(delta)

   # At least one cluster
   knee = max(knee, 1)

   #plt.figure()
   #plt.plot(x,y,'r')
   #plt.show()
   #plt.plot(x,delta, 'b')
   #plt.show()
   #plt.figure()
   #plt.plot(lam, 'r')
   return knee

def getDegreeAndInvSqrt(M):
    D  = np.zeros(M.shape[0])
    DI = np.zeros(M.shape[0])
    DISqrt = np.zeros(M.shape[0])
    r =  np.zeros(M.shape[0], dtype=int)

    rowsum =  sps.csr_matrix.sum(M,1)    
    for i in range(M.shape[0]):
        r[i] = i
        if rowsum[i] >0:
            D[i] = rowsum[i]
            DI[i] = 1/rowsum[i]
        DISqrt[i] = np.sqrt(DI[i])
    

    D  = sps.csr_matrix((D, (r,r)), shape=(M.shape))
    DI = sps.csr_matrix((DI, (r,r)), shape=(M.shape))
    DISqrt = sps.csr_matrix((DISqrt, (r,r)), shape=(M.shape))    
    return D,DI,DISqrt


def computeSymmetricLaplacian(A):
   ## compute the degree matrix, its inverse and inverse square root
   D,DI, DI_sqrt = getDegreeAndInvSqrt(A)
   ## compute laplacian   
   L = D-A 
   ## compute symmetric normalized Laplacian
   Lsym = DI_sqrt*L*DI_sqrt
   return Lsym, D, DI_sqrt
   
def computeRandomWalkLaplacian(A):
    ## compute the degree matrix, its inverse and inverse square root
   D,DI, DI_sqrt = getDegreeAndInvSqrt(A)
   ## compute Random walk Laplacian   
   L = D-A 
   Lrw = DI*L
   return Lrw, D, DI_sqrt
    
def createSparseMatrix(simMeasure, makesym=True):
   r = []  # row indexes
   c = []  # column indexes
   v = []  # Similarity scores 
   nodes_in_graph = len(simMeasure)
   for n in range(nodes_in_graph):
       node_index = n
       for key, value in simMeasure[n]:
           nbr_index = key
           similarity = value
           r.append(node_index)
           c.append(nbr_index)
           v.append(similarity)
       
   r = np.asarray(r, dtype=int)
   c = np.asarray(c, dtype=int)
   v = np.asarray(v, dtype= 'float')
   if len(r) > 0 and len(c) > 0 and len(v) > 0:
   # exponential transform
       vmax  = np.max(v)
       if vmax != 0.0:	   
           v = np.exp(v/vmax)   
       S = sps.csr_matrix((v, (r,c)), shape=(np.max(r)+1,np.max(r)+1))
       if makesym:
           ## make the matrix symetric
           print 'making the matrix symmetric'
           A = S + np.transpose(S)
       else:
           A = S
   else:
       A = None   
   print 'done creating sparse matrix'
   #plt.figure()
   #plt.imshow(S.todense())   
   #plt.show()
   return A, r,c,v

def data_read(sample_dir, json_file):
    namelist = []
    featurelist = []
    
    print 'start reading data'
    start_time = time.time()
    reader = open(json_file, 'r')
    data = json.load(reader)
    imagelist = data["images"]
    listsize = len(imagelist)
    
    # decompose image and feature list
    for i in range(listsize):
        namelist.append(imagelist[i]["name"])
        featurelist.append(imagelist[i]["feature"])
    for i in range(listsize):
        namelist[i] = namelist[i];
        
    end_time = time.time()
    print 'done with data read in %f seconds'%(end_time-start_time)        
    print 'number of images = ', listsize   
    return namelist, featurelist  
    
def data_segment(featurelist, page_size, top_k, diff_thresh, threshMethod, clusteringType):
    print 'data segment()'
    randomSeed = None
    similarity = Similarity();
    _labels = []
    # similarity by nearest neighbor
    nsamples = len(featurelist)
    k = page_size # 40
    l = top_k #5
    threshold = diff_thresh #0.5
    myindex = IndexFlann(k, l, threshold);
    ntype=threshMethod #'byValue' #'byNumber' #
    
    for i in range(nsamples): #range(listsize):
        myindex.addFeatures(featurelist[i]);
        
    print 'sparse matrix computing'
    start_time = time.time()        
    #similarity_measure = myindex.nearestneighbor(ntype);
    similarity_measure = myindex.nearestneighbor_cv(ntype);
    S, r,c,v = createSparseMatrix(similarity_measure, False)
    
    Lsym,D, DI_sqrt = computeSymmetricLaplacian(S);
    end_time = time.time()
    print 'done sparse matrix computing in %f seconds'%(end_time-start_time)
    
    ## pick the first 300 eigenvalues if it is less than the total number of items in the dataset
    #M = np.minimum(400, np.max(r))
    M = np.minimum(300, np.max(r))
    print 'computing eigenpairs'
    start_time = time.time()
    ## call ARPACK routine to compute top k eigenpairs of sparse symmetric matrix  
    lam, v = eigsh(Lsym, M, which='SM' )
    end_time = time.time()
    print 'done computing eigenpairs in %f seconds'%(end_time-start_time)
    #plt.figure()
    #plt.plot(lam)
    #plt.show()
    ## find the "knee" in the the eigenvalue curve to decide on number of clusters
    NC = getClustersBasedonKnee(lam)
    ## create the data matrix fo K-means
    print 'numcluster = %d'%NC
    Data = v[:,0:NC]   # number of points x number of dimensions
    #row normalize D
    Data = rowNormalize(Data)     
    clusters = {}
    
    if clusteringType == 'kmeans':
      
        kmeans = KMeans(init='k-means++', n_clusters=NC, n_init=10, random_state=randomSeed)
        print 'doing kmeans'
        start_time = time.time()
        kmeans.fit(Data)
        _labels = kmeans.predict(Data)
        cluster_centers = kmeans.cluster_centers_
        end_time = time.time()
        print 'done with k means in %f seconds'%(end_time-start_time)
        #return labels, cluster_centers, Data
    elif clusteringType == 'kmedoids':      
        D = spd.cdist(Data, Data)
        print 'doing K-medoids'
        start_time = time.time()
        labels_medoid_ids, medoid_ids = kmedoids.cluster_kmedoid(D, NC)
        unique_labels = np.sort(np.unique(labels_medoid_ids))
        medoid_id_to_cluster_id = {} 
        cluster_id = 0   
        for ul in unique_labels:
            medoid_id_to_cluster_id[ul] = cluster_id
            cluster_id = cluster_id+1
        end_time = time.time()
        print 'done with k medoids in %f seondds'%(end_time-start_time)
        medoid_ids = np.sort(medoid_ids)
        medoids = Data[medoid_ids,:]
        _labels =[]
        for lab in labels_medoid_ids:
            _labels.append(medoid_id_to_cluster_id[lab])
        
        _labels = np.asarray(_labels, dtype=int) 
        
    cluster_list = range(NC)
    clusters = dict.fromkeys(cluster_list, [])
    i = 0
    for indx in _labels:
        clusters[indx] = clusters[indx] + [i]
        i = i + 1      
    
    _keyimage_list = {}
    std = {}
    for key, value in clusters.items():
        flist = []
        cluster_index = key
        cluster_center = cluster_centers[cluster_index]
        for index in value:
            v = Data[index]
            flist = flist + [v]
        diff = flist - cluster_center   
        dist = [np.linalg.norm(v) for v in diff]
        std[cluster_index] = sqrt(np.sum(np.array(dist)*np.array(dist))/len(dist));
        print 'index= ', cluster_index, 'std= ', std[cluster_index]
        x = argmin(dist)
        _keyimage_list[cluster_index] = value[x]
    #return labels,           
    return _labels, _keyimage_list


def data_hierarchy_segment(featurelist, page_size, top_k, diff_thresh, threshMethod, clusteringType, std_thresh, deviate_ratio_thresh, eigen_num=300, label_offset=0):
    print 'data_hierarchy_segment'
    randomSeed = None
    similarity = Similarity();
    _labels = []
    # similarity by nearest neighbor
    nsamples = len(featurelist)
    k = page_size # 40
    l = top_k #5
    threshold = diff_thresh #0.5
    myindex = IndexFlann(k, l, threshold);
    ntype=threshMethod #'byValue' #'byNumber' #
    
    for i in range(nsamples): #range(listsize):
        myindex.addFeatures(featurelist[i]);
        
    print 'sparse matrix computing'
    start_time = time.time()        
    #similarity_measure = myindex.nearestneighbor(ntype);
    similarity_measure = myindex.nearestneighbor_cv(ntype);
    S, r,c,v = createSparseMatrix(similarity_measure, False)
    
    Lsym,D, DI_sqrt = computeSymmetricLaplacian(S);
    end_time = time.time()
    print 'done sparse matrix computing in %f seconds'%(end_time-start_time)
    
    ## pick the first 300 eigenvalues if it is less than the total number of items in the dataset
    #M = np.minimum(400, np.max(r))
    M = np.minimum(eigen_num, np.max(r))
    print 'computing eigenpairs'
    start_time = time.time()
    ## call ARPACK routine to compute top k eigenpairs of sparse symmetric matrix  
    lam, v = eigsh(Lsym, M, which='SM' )
    end_time = time.time()
    print 'done computing eigenpairs in %f seconds'%(end_time-start_time)
    #plt.figure()
    #plt.plot(lam)
    #plt.show()
    ## find the "knee" in the the eigenvalue curve to decide on number of clusters
    NC = getClustersBasedonKnee(lam)
    ## create the data matrix fo K-means
    print 'numcluster = %d'%NC
    Data = v[:,0:NC]   # number of points x number of dimensions
    #row normalize D
    Data = rowNormalize(Data)     
    _clusters = {}
    
    if clusteringType == 'kmeans':
      
        kmeans = KMeans(init='k-means++', n_clusters=NC, n_init=10, random_state=randomSeed)
        print 'doing kmeans'
        start_time = time.time()
        kmeans.fit(Data)
        _labels = kmeans.predict(Data)
        cluster_centers = kmeans.cluster_centers_
        end_time = time.time()
        print 'done with k means in %f seconds'%(end_time-start_time)
        #return labels, cluster_centers, Data
    elif clusteringType == 'kmedoids':      
        D = spd.cdist(Data, Data)
        print 'doing K-medoids'
        start_time = time.time()
        labels_medoid_ids, medoid_ids = kmedoids.cluster_kmedoid(D, NC)
        unique_labels = np.sort(np.unique(labels_medoid_ids))
        medoid_id_to_cluster_id = {} 
        cluster_id = 0   
        for ul in unique_labels:
            medoid_id_to_cluster_id[ul] = cluster_id
            cluster_id = cluster_id+1
        end_time = time.time()
        print 'done with k medoids in %f seondds'%(end_time-start_time)
        medoid_ids = np.sort(medoid_ids)
        medoids = Data[medoid_ids,:]
        _labels =[]
        for lab in labels_medoid_ids:
            _labels.append(medoid_id_to_cluster_id[lab])
        
        _labels = np.asarray(_labels, dtype=int) 
        
    _labels = _labels + label_offset; 
    max_label = np.max(_labels);
    cluster_list = np.array(range(NC)) + label_offset;
    _clusters = dict.fromkeys(cluster_list, [])
    i = 0
    for indx in _labels:
        _clusters[indx] = _clusters[indx] + [i]
        i = i + 1      
    
    _cluster_feature_list = {}
    _std_list = {}
    _dist_list = {}
    _keyimage_list = {}    
    for key, value in _clusters.items():
        features = []
        flist = []
        cluster_index = key
        cluster_center = cluster_centers[cluster_index-label_offset]
        for index in value:
            v = Data[index]
            flist = flist + [v]
            u = featurelist[index]
            features.append(u);
        _cluster_feature_list[key] = features;    
        diff = flist - cluster_center;   
        dist = [np.linalg.norm(v) for v in diff]
        std = sqrt(np.sum(np.array(dist)*np.array(dist))/len(dist));
        dist = np.sort(dist);
        x = argmin(dist)
        _keyimage_list[cluster_index] = value[x]        
        _std_list[key] = std;
        _dist_list[key] = dist;
        print 'index= ', cluster_index, 'size= ', len(value), 'std= ', std, 'min= ',dist[0], 'max= ', dist[len(dist)-1]; 
   
    for key, value in _clusters.items():
       features = _cluster_feature_list[key];
       std = _std_list[key]
       dist = _dist_list[key]
       if std > std_thresh or dist[len(dist)-1] > deviate_ratio_thresh*std:
           labels2,clusters2,keyimage_list2,cluster_feature_list2 = data_hierarchy_segment(features, page_size, 32, 0.8, threshMethod, clusteringType, std_thresh, deviate_ratio_thresh, 100, max_label+1);
           if len(labels2) > 0:
               # update _labels
               for i in range(len(labels2)):
                   _labels[value[i]] = labels2[i]; 
               # update _clusters
               _clusters.pop(key, None)
               for k, v in clusters2.items():
                   u = []
                   for i in v:
                       u = u + [value[i]];
                   clusters2[k] = u;
               _clusters = dict(_clusters.items() + clusters2.items());

               # update _cluster_feature_list
               max_label = np.max(_labels);
               _cluster_feature_list.pop(key, None)
               _cluster_feature_list = dict(_cluster_feature_list.items() + cluster_feature_list2.items())

               # update _keyimage_list
               _keyimage_list.pop(key, None)
               _keyimage_list = dict(_keyimage_list.items() + keyimage_list2.items())
    #return labels,           
    return _labels, _clusters, _keyimage_list, _cluster_feature_list

    
# write all clusters to output folders
def outputClusters_1(_name_list, _label_list, _keyimage_list, sample_dir, output_folder):
    # number of clusters
    print 'write clustered images to ', output_folder
    k = len(_keyimage_list)
    for i in range(k):
        adir = output_folder + str(i)
        if( not os.path.exists(adir)):
            os.mkdir(adir);
    for i in range(len(_name_list)):
        alabel = _label_list[i]
        #print i, alabel
        a, b = _name_list[i].split('_')
        aframe = 'frame' + a
        aname = sample_dir + aframe + '/' + _name_list[i]
        afile = output_folder + str(alabel) + '/' + _name_list[i]
        img = cv2.imread(aname)
        cv2.imwrite(afile, img)   
        
# write all clusters to output folders
def outputClusters_2(_name_list, _label_list, _keyimage_list, sample_dir, output_folder):
    # number of clusters
    if len(_keyimage_list) < 1:
        print 'no cluster is found'
        return
        
    for key, value in _keyimage_list.items():
        adir = output_folder + '/' + str(key)
        if( not os.path.exists(adir)):
            os.mkdir(adir);
    for i in range(len(_name_list)):
        alabel = _label_list[i]
        #print i, alabel
        a, b = _name_list[i].split('_')
        aframe = 'frame' + a
        aname = sample_dir + '/' + aframe + '/' + _name_list[i]
        afile = output_folder + '/' + str(alabel) + '/' + _name_list[i]
        img = cv2.imread(aname)
        cv2.imwrite(afile, img)   

# write all clusters to output folders
def outputClusters_3(_name_list, _label_list, _keyimage_list, sample_dir, output_folder, output_keys):
    # number of clusters
    if len(_keyimage_list) < 1:
        print 'no cluster is found'
        return;
        
    for key, value in _keyimage_list.items():
        adir = output_folder + '/' + str(key)
        if( not os.path.exists(adir)):
            os.mkdir(adir);
        kdir = output_keys + '/key' + str(key)
        if( not os.path.exists(kdir)):
            os.mkdir(kdir);   
       
    # clusters in each folder
    for i in range(len(_name_list)):
        alabel = _label_list[i]
        #print i, alabel
        a, b = _name_list[i].split('_')
        aframe = 'frame' + a
        aname = sample_dir + '/' + aframe + '/' + _name_list[i]
        afile = output_folder + '/' + str(alabel) + '/' + _name_list[i]
        img = cv2.imread(aname)
        cv2.imwrite(afile, img)   

    # key images in each folder
    for key, value in _keyimage_list.items():
        a, b = _name_list[value].split('_')
        kframe = 'frame' + a
        kname = sample_dir + '/' + kframe + '/' + _name_list[value]
        afile = output_keys + '/key' + str(key) + '/' + _name_list[value]
        img = cv2.imread(kname)
        cv2.imwrite(afile, img)   
         
def example_1():                 
    sample_dir_ = '/media/dzhang/CorticalProcessor/data/MBARI/active_learning/usf/00161/All/images/';
    file_dir = '/media/dzhang/CorticalProcessor/data/MBARI/active_learning/usf/00161/All/';
                       
    output_folder_ = file_dir + 'sc/'
    json_file_ = file_dir + 'features.json'

    clusteringType = 'kmeans'; # 'kmedoids'
    threshMethod = 'byValue' #'byValue' # 'byNumber'    
    page_size = 3000
    top_k = 100  #20
    diff_thresh = 0.6
    
    namelist_, featurelist_ = data_read(sample_dir_, json_file_)
    label_index = [x for x in range(len(namelist_))]
    label_list_, keyimage_list_ = data_segment(featurelist_, page_size, top_k, diff_thresh, threshMethod, clusteringType)

    outputClusters_1(namelist_, label_list_, keyimage_list_, sample_dir_, output_folder_)
    print  'done'
    
def example_2():
    video_example = 'testvideo2' #'testvideo1'
    
    sample_dir_ = '/data0/flaskfs/dataset/' + video_example + '/selected/All/images/';
    file_dir = '/data0/flaskfs/dataset/' + video_example + '/selected/All/';
                       
    output_folder_ = file_dir + 'sc/'
    output_keys_ = file_dir + 'keys/'
    json_file_ = file_dir + 'features.json'

    clusteringType = 'kmeans'; # 'kmedoids'
    threshMethod = 'byValue' #'byValue' # 'byNumber'
    page_size = 3000
    top_k = 100  #20
    diff_thresh = 0.6
    eigen_num = 300
    label_offset = 0
    std_thresh = 0.55
    deviate_ratio_thresh = 2.0;
    
    namelist_, featurelist_ = data_read(sample_dir_, json_file_)
    label_list_, clusters_, keyimage_list_, cluster_feature_list_ = data_hierarchy_segment(featurelist_, page_size, top_k, 
                                                                           diff_thresh, threshMethod, clusteringType,
                                                                           std_thresh, deviate_ratio_thresh,
                                                                           eigen_num, label_offset);
    outputClusters_3(namelist_, label_list_, keyimage_list_, sample_dir_, output_folder_, output_keys_)
    print  'done'

def example_3():
    video_example = 'testvideo2' #'testvideo1'
    
    sample_dir_ = '/media/dzhang/CorticalProcessor/data/MBARI/MBARI_cluster_tests/' + video_example + '/images/';
    file_dir = '/media/dzhang/CorticalProcessor/data/MBARI/MBARI_cluster_tests/' + video_example + '/';
                       
    output_folder_ = file_dir + 'sc/'
    output_keys_ = file_dir + 'keys/'
    json_file_ = file_dir + 'features.json'

    clusteringType = 'kmeans'; # 'kmedoids'
    threshMethod = 'byValue' #'byValue' # 'byNumber'
    page_size = 3000
    top_k = 100  #20
    diff_thresh = 0.6
    eigen_num = 300
    label_offset = 0
    std_thresh = 0.55
    deviate_ratio_thresh = 2.0;
    
    namelist_, featurelist_ = data_read(sample_dir_, json_file_)
    label_list_, clusters_, keyimage_list_, cluster_feature_list_ = data_hierarchy_segment(featurelist_, page_size, top_k, 
                                                                           diff_thresh, threshMethod, clusteringType,
                                                                           std_thresh, deviate_ratio_thresh,
                                                                           eigen_num, label_offset);
    outputClusters_3(namelist_, label_list_, keyimage_list_, sample_dir_, output_folder_, output_keys_)
    print  'done'

def example_4():
    video_example = 'testvideo2' #'testvideo1'
    
    sample_dir_ = '/media/dzhang/CorticalProcessor/data/MBARI/USF_dataset/USF_SD_LinStrtch/All/images/';
    file_dir = '/media/dzhang/CorticalProcessor/data/MBARI/USF_dataset/USF_SD_LinStrtch/All/';
                       
    output_folder_ = file_dir + 'sc/'
    output_keys_ = file_dir + 'keys/'
    json_file_ = file_dir + 'features.json'

    clusteringType = 'kmeans'; # 'kmedoids'
    threshMethod = 'byValue' #'byValue' # 'byNumber'
    page_size = 3000
    top_k = 100  #20
    diff_thresh = 0.6
    eigen_num = 300
    label_offset = 0
    std_thresh = 0.55
    deviate_ratio_thresh = 2.0;
    
    namelist_, featurelist_ = data_read(sample_dir_, json_file_)
    label_list_, clusters_, keyimage_list_, cluster_feature_list_ = data_hierarchy_segment(featurelist_, page_size, top_k, 
                                                                           diff_thresh, threshMethod, clusteringType,
                                                                           std_thresh, deviate_ratio_thresh,
                                                                           eigen_num, label_offset);
    outputClusters_3(namelist_, label_list_, keyimage_list_, sample_dir_, output_folder_, output_keys_)
    print  'done'
    
if __name__ == '__main__':
    example_2()
