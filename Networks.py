import numpy as np
import copy
import time as t
class FFNNetwork:
  def __init__(self, layers, errorfunc):
    self.layers = layers
    self.inputsize = layers[0].inputsize
    self.outputsize = layers[-1].outputsize
    self.errorfunc = errorfunc
    self.gradients = []
    self.output = None
    for i in self.layers:
      self.gradients.append(i.gradients)
  def forward(self, input):
    self.input = input
    for e in self.layers:
      #time = t.time()
      self.input = e.forward(self.input)
      #print(round(t.time() - time, 3), e)
    self.output = self.input
    return self.output
  def backward(self, doutput, learnrate):
    self.doutput = doutput
    self.gradients = []
    for e in reversed(self.layers):
      #time = t.time()
      self.doutput = e.backward(self.doutput, learnrate)
      self.gradients.insert(0, e.gradients)
      #print(round(t.time() - time, 3), e)
    return self.doutput
  def train(self, x, y, epochs, learnrate, learnratemultiplyer = float(1), shuffle = True, EpochsPerReport = 100, Report = False, Batchsize = 1000, Batchs = False):
    prex = x
    prey = y
    time = 0
    self.learnrate = learnrate
    Correct = 0
    TotalError = 0
    RoundsRecorded = 0
    print("Training Started")
    for i in range(epochs):
      if shuffle:
        p = np.random.permutation(len(prex))
        prex = prex[p]
        prey = prey[p]
      if Batchs:
        x = prex[:Batchsize]
        y = prey[:Batchsize]
      else:
        x = prex
        y = prey
      TotalError = 0
      Correct = 0
      RoundsRecorded = 0
      for e in range(len(x)):
        if i == 0 and e == 0:
          time = t.time()
        output = self.forward(x[e])
        if Report:
          Error = self.errorfunc.forward(output, y[e])
          TotalError += Error
          Correct += int(np.argmax(output) == np.argmax(y[e]))
          RoundsRecorded += 1
        self.backward(self.errorfunc.backward(output, y[e]), self.learnrate)
        if i == 0 and e == 9:
          totalsingletime = (t.time() - time) / 10
          totaltime = totalsingletime * len(x) * epochs
          hours = totaltime // 3600
          minutes = (totaltime % 3600) // 60
          seconds = totaltime % 60
          print('Estimated training time: ', int(hours), ' hours, ', int(minutes), ' minutes, ', int(seconds), ' seconds')
        self.learnrate = self.learnrate * (learnratemultiplyer ** (1 / len(x)))
      if (i + 1) % EpochsPerReport == 0 and Report:
        print('Epoch ', i+1, ', ', round((Correct / RoundsRecorded) * 100, 3),"% Correct, Error: ", round(TotalError / RoundsRecorded, 5))
    print("Training Done")
  def test(self, x, y):
    Correct = 0
    TotalError = 0
    for i in range(len(x)):
      output = self.forward(x[i])
      Error = self.errorfunc.forward(output, y[i])
      TotalError += Error
      Correct += int(np.argmax(output) == np.argmax(y[i]))
    print('Test Done \n', round((Correct / len(x)) * 100, 3), '% Correct, Error: ', round(TotalError / len(x), 5))
  def adjust(self, adjustment):
    for i in range(len(self.layers)):
      self.layers[i].adjust(adjustment[i])
class RNNetwork:
  def __init__(self, inputlayer, memorylayer, inputmemorylayer, outputlayer, errorfunc, oneoutput = True):
    self.inputsize = inputlayer.inputsize
    self.outputsize = outputlayer.outputsize
    self.inputlayer = inputlayer
    self.memorylayer = memorylayer
    self.oneoutput = oneoutput
    self.inputmemorylayer = inputmemorylayer
    self.outputlayer = outputlayer
    self.errorfunc = errorfunc
    self.memory = np.zeros((self.memorylayer.inputsize, 1))
    self.memorys = []
    self.outputs = []
    self.rolledoutnetwork = []
  def singleforward(self, input, memory):
    self.input = input
    self.memory = memory
    i = self.inputlayer.forward(self.input)
    m = self.memorylayer.forward(self.memory)
    output = self.inputmemorylayer.forward(i + m)
    if self.oneoutput:
      return output
    else:
      trueout = self.outputlayer.forward(output)
      return output, trueout
  def singlebackward(self, savednetwork, dmemory, learnrate, doutput = None, memory = True):
    self.doutput = doutput
    self.dmemory = dmemory
    if not(self.oneoutput):
      if memory:
        self.dmemory += savednetwork[3].backward(self.doutput, learnrate)
      else:
        self.dmemory = savednetwork[3].backward(self.doutput, learnrate)
    self.doutput = self.dmemory
    self.doutput = savednetwork[2].backward(self.doutput, learnrate)
    savednetwork[0].backward(self.doutput, learnrate)
    self.doutput = savednetwork[1].backward(self.doutput, learnrate)
    return self.doutput, (savednetwork[0].gradients, savednetwork[1].gradients, savednetwork[2].gradients, savednetwork[3].gradients)
  def forward(self, input):
    self.rolledoutnetwork = []
    self.outputs = []
    for i in range(len(input)):
      self.memorys.append(self.memory)
      if self.oneoutput:
        self.memory = self.singleforward(input[i], self.memory)
        if i == len(input) - 1:
          self.output = self.outputlayer.forward(self.memory)
      else:
        self.memory, self.output = self.singleforward(input[i], self.memory)
        self.outputs.append(self.output)
      self.rolledoutnetwork.append( copy.deepcopy([self.inputlayer, self.memorylayer, self.inputmemorylayer, self.outputlayer]))
    if self.oneoutput:
      return self.output
    else:
      return self.outputs
  def backward(self, doutputs, learnrate):
    self.doutputs = doutputs
    self.gradients = []
    self.doutput = doutputs[-1]
    for i in reversed(range(len(self.rolledoutnetwork))):
      if self.oneoutput:
        if i == len(self.rolledoutnetwork) - 1:
          self.doutput = self.rolledoutnetwork[i][3].backward(self.doutputs, learnrate)
          self.outputlayer.adjust(self.rolledoutnetwork[i][3].gradients)
        self.doutput, self.gradients = self.singlebackward(self.rolledoutnetwork[i], self.doutput, learnrate)
      else: 
        if i == len(self.rolledoutnetwork) - 1:
          self.doutput, self.gradients = self.singlebackward(self.rolledoutnetwork[i], self.doutput, learnrate, self.doutputs[i], False)
      self.inputlayer.adjust(self.gradients[0])
      self.memorylayer.adjust(self.gradients[1])
      self.inputmemorylayer.adjust(self.gradients[2])
      if not(self.oneoutput):
        self.outputlayer.adjust(self.gradients[3])
    return self.doutput
  def train(self, x, y, epochs, learnrate, shuffle = True, EpochsPerReport = 100, Report = False, Batchsize = 1000, Batchs = False):
    prex = x
    prey = y
    Correct = 0
    TotalError = 0
    Totaloutputs = 0
    RoundsRecorded = 0
    for i in range(epochs):
      if shuffle:
        p = np.random.permutation(len(prex))
        prex = prex[p]
        prey = prey[p]
      if Batchs:
        x = prex[:Batchsize]
        y = prey[:Batchsize]
      else:
        x = prex
        y = prey
      if (i + 1) % EpochsPerReport == 0 and Report:
        print('Epoch ', i+1, ', ', round((Correct / Totaloutputs) * 100, 3), '% Correct Error: ', round(TotalError / RoundsRecorded, 5))
        TotalError = 0
        Correct = 0
        Totaloutputs = 0
        RoundsRecorded = 0
      for e in range(len(x)):
        output = self.forward(x[e])
        doutput = []
        if Report:
          preError = []
          for j in range(len(output)):
            preError.append(self.errorfunc.forward(output[j], y[e][j]))
            Correct += int(np.argmax(output[j]) == np.argmax(y[e][j]))
            Totaloutputs += 1
          Error = np.sum(preError)/len(preError)
          TotalError += Error
          RoundsRecorded += 1
        for j in range(len(output)):
          doutput.append(self.errorfunc.backward(output[j], y[e][j]))
        self.backward(doutput,learnrate)
    print("Training Done")
  def test(self, x, y):
    TotalError = 0
    Correct = 0
    Totaloutputs = 0
    for i in range(len(x)):
      output = self.forward(x[i])
      preError = []
      for j in range(len(output)):
        preError.append(self.errorfunc.forward(output[j], y[i][j]))
        Correct += int(np.argmax(output[j]) == np.argmax(y[i][j]))
        Totaloutputs += 1
      Error = np.sum(preError)/len(preError)
      TotalError += Error
    print('Test Done \n', round((Correct / Totaloutputs) * 100, 3), '% Correct, Error: ', round(TotalError / len(x), 5))
  def adjust(self, adjustment):
    for i in range(len(self.inputlayer)):
      self.inputlayer[i].adjust(adjustment[i])
    for i in range(len(self.outputlayer)):
      self.outputlayer[i].adjust(adjustment[i])
    for i in range(len(self.memorylayer)):
      self.memorylayer[i].adjust(adjustment[i])
    for i in range(len(self.inputmemorylayer)):
      self.inputmemorylayer[i].adjust(adjustment[i])