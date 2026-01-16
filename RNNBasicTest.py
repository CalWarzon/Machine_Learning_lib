import numpy as np
import Networks as N
import MainLayers as L
import ActivationLayers as AL
from ErrorFunctions import MeanSquaredError as MSE
x = np. reshape([[[0, 1],[1, 0],[1, 0],[0, 1],[1, 0],[1, 0]],[[1, 0],[0, 1],[1, 0],[1, 0],[0, 1],[1, 0]],[[1, 0],[1, 0],[0, 1],[1, 0],[1, 0], [0, 1]]], (3,6,2,1))
y = np. reshape(([[1],[1],[0],[1],[1],[0]], [[0],[1],[1],[0],[1],[1]],[[1],[0],[1],[1],[0],[1]]), (3,6,1))
RNN = N.RNNetwork(L.Denselayer(2,4), L.Denselayer(2,4), N.FFNNetwork((L.Denselayer(4,8), AL.Tanh(8), L.Denselayer(8,2),AL.Tanh(2)),MSE()), N.FFNNetwork((L.Denselayer(2,1), AL.Sigmoid(1)),MSE()), MSE(), True)
RNN.train(x, y, 1000, 0.1, True, 100, True, 1000, False)
RNN.test(x, y)