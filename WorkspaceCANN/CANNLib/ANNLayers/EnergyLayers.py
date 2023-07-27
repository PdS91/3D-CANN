# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:34:55 2023

@author: Pietro
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class PsiComputer(torch.nn.Module):

  def __init__(self,R=5, *args, **kwargs) -> None:
      super(PsiComputer,self).__init__(*args, **kwargs)
      self.R = R

      self.main = nn.Sequential(
          
          nn.Flatten(),
          nn.Linear(R*2,R),
          nn.Tanh(),
          nn.Linear(R,R),
          nn.Tanh(),
          nn.Linear(R,1),
          nn.ReLU()
        
      )

  def forward(self,f_):

    Psi_ = self.main(f_)

    return Psi_
