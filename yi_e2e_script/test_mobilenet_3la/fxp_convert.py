import numpy
import torch 

#torch.set_printoptions(precision=6)

def quantize_(x, qi, qf):
  fmax = 1. - float(torch.pow(2., torch.FloatTensor([-1. * qf])).numpy()[0])
  imax =      float((torch.pow(2., torch.FloatTensor([qi-1])) - 1).numpy()[0])
  imin = -1 * float((torch.pow(2., torch.FloatTensor([qi-1]))).numpy()[0])
  fdiv = float(torch.pow(2., torch.FloatTensor([-qf])).numpy()[0])

  x = torch.floor ( x / fdiv) * fdiv
  x = torch.clamp(x, imin, imax + fmax)

  #print (x)
  return x