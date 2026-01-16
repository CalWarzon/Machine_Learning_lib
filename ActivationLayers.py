import numpy as np
import BaseActivation
class Tanh(BaseActivation.Activation):
  def __init__(self, size):
    super().__init__(lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2, size)
class Sigmoid(BaseActivation.Activation):
  def __init__(self, size):
    sigmoid = lambda x: 1 / (1 + np.exp(-(x-np.max(x))))
    super().__init__(sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x)), size)
class ReLU(BaseActivation.Activation):
  def __init__(self, size):
    super().__init__(lambda x: np.maximum(0, x), lambda x: np.where(x > 0, 1, 0), size)
class Softmax(BaseActivation.Activation):
  def __init__(self, size):
    softmax = lambda x: np.exp(x - np.max(x))/np.sum(np.exp(x - np.max(x)))
    super().__init__(softmax, lambda x: softmax(x) * (1 - softmax(x)), size)