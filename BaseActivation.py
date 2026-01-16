import numpy as np
import BaseLayer
class Activation(BaseLayer.Layer):
  def __init__(self, func, dfunc, size):
    self.func = func
    self.dfunc = dfunc
    self.inputsize = size
    self.outputsize = size
    self.output = None
    self.gradients = []
  def forward(self, input):
    self.input = input
    self.output = self.func(input)
    return self.output
  def backward(self, doutput, learnrate): 
    return np.multiply(doutput, self.dfunc(self.input))
  def adjust(self, adjustment):
    pass