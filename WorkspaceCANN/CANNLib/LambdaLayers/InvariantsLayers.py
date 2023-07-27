# -*- coding: utf-8 -*-
"""
Created on Sat May 13 15:41:01 2023

@author: Pietro
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class GeneralizedInvariantsComputer(torch.nn.Module):

  def __init__(self, *args, **kwargs) -> None:
      super(GeneralizedInvariantsComputer,self).__init__(*args, **kwargs)


  def forward(self,Lgen_,C_):

    IJgen_ = self.IJgen_Computation_Vec(Lgen_,C_)

    return IJgen_

  def IJgen_Computation_Vec(self,Lgen_,C_):

    # C_ --> deformation tensor [?,3,3]
    # Lgen_ --> generalized structural tensor vector [?,R,3,3]

    Igen = lambda C,Lgen : torch.trace(torch.matmul(C,Lgen))
    Jgen = lambda dC,Lgen : torch.trace(dC*Lgen)

    def Compute_dC (C):
      
      Cit = torch.transpose(torch.inverse(C),0,1)

      dC = torch.det(torch.matmul(C,Cit))

      return dC

    def Compute (Lgenset):

      comp_dC = torch.vmap(Compute_dC)
      dC_ = comp_dC(C_)

      comp_Igen = torch.vmap(Igen)
      comp_Jgen = torch.vmap(Jgen)

      Igen_ = comp_Igen(C_,Lgenset)
      Jgen_ = comp_Jgen(dC_,Lgenset)

      return Igen_,Jgen_

    vectorized = torch.vmap(Compute,in_dims=1)

    I_,J_ = vectorized(Lgen_)

    I_ = torch.transpose(I_,1,0)
    J_ = torch.transpose(J_,1,0)

    I_ = torch.unsqueeze(I_,2)
    J_ = torch.unsqueeze(J_,2)

    IJ_ = torch.concatenate((I_,J_),axis=2)

    return IJ_




class GeneralizedStructuralTensorComputer(torch.nn.Module):

  def __init__(self, *args, **kwargs) -> None:
      super(GeneralizedStructuralTensorComputer,self).__init__(*args, **kwargs)


  def forward(self,L_,w_):

    Lgen_ = self.Lgen_Computation_Vec(L_,w_)

    return Lgen_


  def Lgen_Computation_Vec(self,L_,w_):

    # w_ --> weight tensor [?,J+1,R]
    # L_ --> structural tensor vector [?,J+1,3,3]

    customdot = lambda w,L : torch.tensordot(w,L,dims=([0],[0]))

    vectorized = torch.vmap(customdot)
    Lgen_ = vectorized(w_,L_)

    return Lgen_



class StructuralTensorComputer(torch.nn.Module):

  def __init__(self, *args, **kwargs) -> None:
      super(StructuralTensorComputer,self).__init__(*args, **kwargs)


  def forward(self,l_):

    L_ = self.L_Computation_Vec(l_)

    return L_


  def L_Computation_Vec(self, l_):

    customdot = lambda l : torch.tensordot(l,l,dims=0)

    vectorized = torch.vmap(torch.vmap(customdot))

    L_ = vectorized(l_)

    I = torch.eye(3,3)*1/3
    I = I.repeat(l_.shape[0],1)
    I = torch.reshape(I,(l_.shape[0],1,3,3))

    L_ = torch.cat((L_,I),1)

    return L_


