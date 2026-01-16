import numpy as np
from numpy._typing import _256Bit
from numpy.core.fromnumeric import argmax
import Networks as N
import MainLayers as L
import ActivationLayers as AL
from ErrorFunctions import MeanSquaredError as MSE
from ErrorFunctions import BinaryCrossEntropy as BCE
import ImportExport as IE
import Utilitys as U
import time as t
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.mnist import load_data as mn
(prextrain, preytrain), (prextest, preytest) = mn()
#prextrain = prextrain/ 255
#prextest = prextest/ 255
xtrain = np.reshape(prextrain, (60000, 1, 28, 28))
ytrain = np.zeros((len(preytrain), 10))
ytrain = np.reshape(ytrain, (60000, 10, 1))
for i in range(len(preytrain)):
  ytrain[i][preytrain[i]] = 1
CNN = IE.importfile('mnistCNN.txt')
out = CNN.forward(xtrain[4234])
print(CNN.layers[0].outputsize)
U.PrintMatrix(prextrain[4234], 200, 150, True)
print(np.round(out, 2))
U.PrintFeatureMap(CNN.layers[0].output, 'gray', 2)

#CNN.train(xtrain, ytrain, 3, 0.001, 0.90, True, 1, True, 1000, True)
#IE.exportfile(CNN, "emnistCNN.txt")