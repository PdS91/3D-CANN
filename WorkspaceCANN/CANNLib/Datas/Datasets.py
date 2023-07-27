# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:55:16 2023

@author: Pietro
"""

import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch

class IsotropicIncompressibleDataset(torch.utils.data.Dataset):
    
  def __init__(self, seed=64, size=100, Fmin = 0.5, Fmax = 2, *args, **kwargs):
    
    self.seed = int(seed)
    self.size = int(size)
    
    self.c10 = 1.6e-1
    self.c20 = -1.4e-3
    self.c30 = 3.9e-3
    
    self.c01 = 1.5e-2
    self.c02 = -2.0e-6
    self.c03 = 1e-10
    
    self._F = np.empty((self.size,3,3))
    self._C = np.empty((self.size,3,3))
    self._features = np.ones((self.size,3))

    self._Psi = np.empty((self.size,1))
    
    self._GenerateCDatas(Fmin = Fmin, Fmax = Fmax)
    self._ComputePsiTarget()
      
    return
    
      

  def __len__(self):

    return self.size

  def __getitem__(self, idx):
      
      C = torch.tensor(self._C[idx], dtype = torch.float32)
      f = torch.tensor(self._features[idx], dtype = torch.float32)

      data = {'C': C, 'f': f}

      Psi = torch.tensor(self._Psi[idx])

      sample = {'Data':data,'Psi':Psi}

      return sample


  def _GenerateCDatas(self, Fmin, Fmax):
    
    generator = np.random.default_rng(self.seed)
    
    #
    Fmin = 0.5
    Fmax = 1.5
    
    Fs = generator.uniform(Fmin,Fmax,(self.size,3,3))
    
    self._F = Fs
    Ft = np.swapaxes(Fs, 1, 2)
    
    
    Cs = Ft@Fs
    
    self._C = Cs

    return

  def _ComputePsiTarget(self):
    
    C = self._C
    
    def Compute_dC (C):
      
      Cit = np.transpose(np.linalg.inv(C))

      dC = np.linalg.det(C@Cit)

      return dC
  
    dC = np.empty(C.shape[0])
    
    for index,c in enumerate(C):
        
        dC[index] = Compute_dC(c)      
    
    Ic = np.trace(C,axis1=1,axis2=2)
    IIc = dC
    
    param = np.array([self.c10,self.c01,self.c20,self.c02,self.c30,self.c03]).reshape((3,2))
    
    Psi=0
    
    for i,j in enumerate(param):
        
        Psi += j[0]*(Ic-3)**(i+1) + j[1]*(IIc-3)**(i+1)
        
        
    self._Psi = Psi
        
    return


                    
                      
      
      
      
      
      
      