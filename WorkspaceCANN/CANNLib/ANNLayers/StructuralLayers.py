import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class DirectionComputer(torch.nn.Module):

  def __init__(self,feature_number=7, J=3, *args, **kwargs) -> None:
      super(DirectionComputer,self).__init__(*args, **kwargs)


      self.main = nn.Sequential(
          
          nn.Linear(feature_number,2*J),
          nn.Tanh(),
          nn.Linear(2*J,2*J),
          nn.Tanh(),
          nn.Linear(2*J,3*J),
          nn.Unflatten(1,(J,3))
        
      )

  def forward(self,f_):

    l = self.main(f_)
    l_ = nn.functional.normalize(l,dim=2)

    return l_


class WeightComputer(torch.nn.Module):

  def __init__(self, feature_number = 7, J = 3, R = 5, *args, **kwargs) -> None:
      super(WeightComputer,self).__init__(*args, **kwargs)

      J = J+1

      self.main = nn.Sequential(
          
          nn.Linear(feature_number, 2*J),
          nn.Tanh(),
          nn.Linear(2*J, 2*J),
          nn.Tanh(),
          nn.Linear(2*J, J*R),
          nn.Unflatten(1,(J, R))
        
      )

  def forward(self,f_):

    w = self.main(f_)
    w = w*w
    w_ = nn.functional.normalize(w,dim=1,p=1)

    return w_



    