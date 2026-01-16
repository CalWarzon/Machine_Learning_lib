import numpy as np
from scipy import signal
import BaseLayer
class DenseLayer(BaseLayer.Layer):
  def __init__(self, inputsize, outputsize):
    self.outputsize = outputsize
    self.inputsize = inputsize
    self.output = None
    self.biases = np.random.randn(outputsize, 1)
    self.weights = np.random.randn(outputsize, inputsize)
    self.gradients = [np.zeros(self.weights.shape), np.zeros(self.biases.shape)]
  def forward(self, inputs):
    self.inputs = inputs
    self.output = np.dot(self.weights, self.inputs) + self.biases
    return self.output
  def backward(self, doutput, learnrate):
    gradbiases = learnrate * doutput
    gradweights = learnrate * np.dot(doutput, self.inputs.transpose())
    gradoutputs = np.dot(self.weights.transpose(), doutput)
    self.gradients = [gradweights, gradbiases]
    self.biases -= gradbiases
    self.weights -= gradweights
    return gradoutputs
  def adjust(self, adjustment):
    self.biases -= adjustment[1]
    self.weights -= adjustment[0]
class ConvolutionalLayer(BaseLayer.Layer):
  def __init__(self, inputsize, kernelsize, depth):
    self.depth = depth
    self.kernelsize = kernelsize
    self.inputsize = inputsize
    self.inputdepth, self.inputhight, self.inputwidth = inputsize
    self.outputsize = (self.depth, self.inputhight - self.kernelsize + 1, self.inputwidth - self.kernelsize + 1)
    self.kernelshape = (self.depth, self.inputdepth, self.kernelsize, self.kernelsize)
    self.kernels = np.random.randn(*self.kernelshape)
    self.biases = np.random.randn(*self.outputsize)
    self.gradients = [np.zeros(self.kernels.shape), np.zeros(self.biases.shape)]
  def forward(self, input):
    self.input = input
    self.output = np.copy(self.biases)
    for i in range(self.depth):
      for j in range(self.inputdepth):
        self.output[i] += signal.correlate2d(self.input[j], self.kernels[i][j], 'valid')
    return self.output
  def backward(self, doutput, learnrate):
    self.gradbiases = learnrate * doutput
    self.gradkernels = np.zeros(self.kernelshape)
    self.gradoutputs = np.zeros(self.inputsize)
    for i in range(self.depth):
      for j in range(self.inputdepth):
        self.gradkernels[i, j] = signal.correlate2d(self.input[j], doutput[i], 'valid')
        self.gradoutputs[j] += signal.convolve2d(doutput[i], self.kernels[i, j], 'full')
    self.gradkernels = learnrate * self.gradkernels
    self.gradients = [self.gradkernels, self.gradbiases]
    self.biases -= self.gradbiases
    self.kernels -= self.gradkernels
    return self.gradoutputs
  def adjust(self, adjustment):
    self.biases -= adjustment[1]
    self.kernels -= adjustment[0]
class ReshapeLayer(BaseLayer.Layer):
  def __init__(self, inputshape, outputshape):
    self.inputsize = inputshape
    self.outputsize = outputshape
    self.gradients = None
  def forward(self, input):
    return np.reshape(input, self.outputsize)
  def backward(self, doutput, learnrate):
    return np.reshape(doutput, self.inputsize)
  def adjust(self, adjustment):
    pass
class MaxPoolingLayer(BaseLayer.Layer):
  def __init__(self, inputsize, filtersize):
    self.inputsize = inputsize
    self.inputpadding = [0,0]
    self.filtersize = filtersize
    self.gradients = None
    self.inputpadding = ((self.filtersize -(self.inputsize[1] % self.filtersize)) * int(self.inputsize[1] % self.filtersize != 0), (self.filtersize - (self.inputsize[2] % self.filtersize)) * int(self.inputsize[2] % self.filtersize != 0))
    self.paddedinputsize = (int(self.inputsize[0]), int(self.inputsize[1] + self.inputpadding[0]), int(self.inputsize[2] + self.inputpadding[1]))
    self.outputsize = (int(self.paddedinputsize[0]), int(self.paddedinputsize[1] / self.filtersize), int(self.paddedinputsize[2] / self.filtersize))
  def forward(self, input):
    self.input = input
    self.output = np.zeros(self.outputsize)
    min = np.min(self.input)
    self.paddedinput = np.pad(self.input, ((0, 0), (0, self.inputpadding[0]), (0, self.inputpadding[1])), 'constant', constant_values = ((min, min), (min, min), (min, min)))
    for i in range(self.outputsize[0]):
      for e in range(self.outputsize[1]):
        for j in range(self.outputsize[2]):
          test = self.paddedinput[i, e * self.filtersize : e * self.filtersize + self.filtersize, j * self.filtersize : j * self.filtersize + self.filtersize]
          self.output[i][e][j] = np.max(test)
    return self.output
  def backward(self, doutput, learnrate):
    self.doutput = doutput
    dinput = np.zeros(self.inputsize)
    for i in range(self.outputsize[0]):
      for e in range(self.outputsize[1]):
        for j in range(self.outputsize[2]):
          for h in range(self.filtersize):
            for k in range(self.filtersize):
              inputlocation = (i, (e * self.filtersize) + h, (j * self.filtersize) + k)
              if self.inputsize[1] > inputlocation[1] and self.inputsize[2] > inputlocation[2]:
                if self.output[i][e][j] == self.input[inputlocation]:
                  dinput[inputlocation] = doutput[i][e][j]
    self.doutput = dinput
    return self.doutput
  def adjust(self, adjustment):
    pass