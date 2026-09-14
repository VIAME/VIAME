# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

import torch
import numpy as np
import scipy
import sys
import pdb

class ReMax:
    #Can accept data as numpy, torch.Tensor, or list
    #Can multiply raw data by scale values
    #Always fits on GT data
    def __init__(self, data, rescale = None, bias = 0, threshold=None): 
        self.rescale = rescale
        self.bias = bias
        
        data = self.conform_data(data) #Get the data into a numpy ndarray
        
        if rescale != None:
            rescale = self.conform_data(rescale)#Get the scales into a numpy ndarray
            data = data * rescale

        self.shape,self.loc,self.scale = scipy.stats.genpareto.fit(data)

    def ReScore(self, data, rescale=None): #Always returns probability of known
        
        data = self.conform_data(data) #Get the data into a numpy ndarray
        
        if rescale != None:
            rescale = self.conform_data(rescale) #Get the scale values into a numpy ndarray
            data = data * rescale

        return torch.from_numpy(scipy.stats.genpareto.cdf(data, self.shape, loc=self.loc, scale=self.scale)) - self.bias
    
    def conform_data(self, data):
        assert type(data) == torch.Tensor or type(data) == list or type(data) == np.ndarray or type(data) == np.float32
        
        if type(data) == torch.Tensor:
            data = data.detach()
            if data.device.type == 'cuda':
                data = data.cpu()
            data = data.numpy()
        elif type(data) == list:
            data = np.asarray(data)
            
        return data.reshape(-1)
    
    def find_threshold(self, data, rescale=None, acceptance_percentage=0.95):
        print("BETA")
        data = self.conform_data(data)
        
        rescored_data = self.ReScore(data, rescale)
        
        sorted_rescore_data = np.sort(rescored_data, axis=0)[::-1]#Sort in reverse order
        
        threshold_index = int(round(sorted_rescore_data.shape[0]*acceptance_percentage, 0))
        self.threshold = sorted_rescore_data[threshold_index]
        
        
        effective_threshold = np.sum(sorted_rescore_data >= self.threshold, axis=0)/sorted_rescore_data.shape[0]
        
        if effective_threshold != self.threshold:
            print(f"Rescore found a threshold ({self.threshold}) to accept {acceptance_percentage}% of validation data, but because of identical ReScores, it actually accepts {effective_threshold}")
        
        return self.threshold
        
