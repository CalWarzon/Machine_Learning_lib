import numpy as np
from MainLayers import DenseLayer
from ActivationLayers import Tanh
from ErrorFunctions import MeanSquaredError
def test():
  xs = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
  ys = np.reshape([[0],[1],[1],[0]], (4,1,1))
  network = [DenseLayer(2,3), Tanh(3), DenseLayer(3,1), Tanh(1)]
  Errorfunc = MeanSquaredError()
  epochs = 2000
  learnrate = 0.1
  for i in range(epochs):
    for j in range(len(xs)): 
      input = xs[j]
      for e in network:
        input = e.forward(input)
      error = Errorfunc.forward(input, ys[j])
      d = Errorfunc.backward(input, ys[j])
      for e in reversed(network):
        d = e.backward(d, learnrate)
  for i in range(len(xs)):
    input = xs[i]
    for e in network:
      input = e.forward(input)
    error = Errorfunc.forward(input, ys[i])
    print(xs[i])
    print(error)
    print(input)
    print()