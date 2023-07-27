import numpy as np
import torch

from CANNLib.ANNLayers.EnergyLayers import *
from CANNLib.ANNLayers.StructuralLayers import *
from CANNLib.LambdaLayers.InvariantsLayers import *


class CANNClassic_PsiRegressor(torch.nn.Module):

  def __init__(self, J = 3, R = 5, f = 3, *args, **kwargs) -> None:
      super(CANNClassic,self).__init__(*args, **kwargs)

      self.DC = DirectionComputer(feature_number = f, J = J)
      self.WC = WeightComputer(feature_number = f, J = J, R = R)
      self.PC = PsiComputer(R = R)

      self.STC = StructuralTensorComputer()
      self.GSTC = GeneralizedStructuralTensorComputer()
      self.GIC = GeneralizedInvariantsComputer()

  def forward(self,data):

    C_ = data['C']
    f_ = data['f']


    l_ = self.DC(f_)
    w_ = self.WC(f_)

    #print(l_,w_)

    L_ = self.STC(l_)
    Lgen_ = self.GSTC(L_,w_)
    IJgen_ = self.GIC(Lgen_,C_)

    #print(L_,Lgen_,IJgen_)

    Psi_ = self.PC(IJgen_)

    return Psi_
