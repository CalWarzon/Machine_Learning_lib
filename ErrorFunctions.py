import numpy as np
class MeanSquaredError:
  def __init__(self):
    pass
  def forward(self, inputs, targets):
    return np.mean(np.square(inputs - targets))
  def backward(self, inputs, targets):
    return 2 * (inputs - targets) / inputs.size
class BinaryCrossEntropy:
  def __init__(self):
    pass
  def forward(self, inputs, targets):
    return -np.mean(targets * np.log(inputs) + (1 - targets) * np.log(1 - inputs))
  def backward(self, inputs, targets):
    return (((1 - targets) / (1 - inputs)) - (targets / inputs)) / np.size(targets)