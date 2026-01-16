import dill as pickle
def exportfile(object, file):
  with open(file, 'wb') as f:
    pickle.dump(object, f)
def importfile(file):
  with open(file, 'rb') as f:
    return pickle.load(f)