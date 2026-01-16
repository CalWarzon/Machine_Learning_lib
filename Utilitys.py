import matplotlib.pyplot as plt
import numpy as np
def PrintMatrix(matrix, visiblethreashhold, invisiblethreashholdforshaden = float(0),shade = False):
  shading = ['  ' , '░░', '▒▒', '▓▓', '██']
  shadestep = (visiblethreashhold - invisiblethreashholdforshaden) / 3
  for i in matrix:
    for e in i:
      if e >= visiblethreashhold:
        print(shading[4], end = '')
      elif shade and invisiblethreashholdforshaden < e:
        print(shading[int((e - invisiblethreashholdforshaden) / shadestep) + 1], end = '')
      else:
        print(shading[0], end = '')
    print('\n', end = '')
def PrintMatrixPyplot(matrix, color = None):
  if color != None:
    plt.imshow(matrix, cmap = color)
  else:
    plt.imshow(matrix)
  plt.show()

def PrintFeatureMapPyplot(Map, color = None, numwidth = 3):
  figure, axes = plt.subplots((len(Map)//numwidth) + int(len(Map)%numwidth != 0), numwidth)
  for i in range(len(Map)):
    if color != None:
      axes[i // numwidth, i % numwidth].imshow(Map[i] ,cmap = color)
    else:
      axes[i // numwidth, i % numwidth].imshow(Map[i])
  plt.show()

def PrintFeatureMap(Map, visiblethreashhold, invisiblethreashholdforshaden = float(0),shade = False):
  for i in range(len(Map)):
    PrintMatrix(map[i], visiblethreashhold, invisiblethreashholdforshaden, shade)
    print('/n')

  