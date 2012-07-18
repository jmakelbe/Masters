#!/usr/bin/python

import sys, getopt
import numpy as np
import scipy.optimize as op
import datetime

from logreg import *
#multitask
homePath = '/Users/July/Documents/Dropbox/OxfordMSc/Thesis/code/'
#homePath = '/cluster/julie/code/'

actors = 5

crossvalid = False
regular = False

# These three functions are helpers for the BFGS optimiser
last_func=0
iterations=0
arrays_of_means=0
arrays_of_std=0

def function(W, D, r1, r2): 
  global last_func
  last_func = multi_objective(W, D, r1, r2)
  last_func = last_func + regularisation(W, r1, r2)
  return last_func

def function_grad(W, D, r1, r2):
  res = grad_multi_objective(W, D, r1, r2)
  return res

def report(w):
  global last_func
  global iterations
  if not (iterations % 10):
    print '   iteration', iterations, ': cross entropy=', last_func
  iterations += 1

def addLabels(data, positive):
  if positive:
    labels = np.ones([data.shape[0],1])
  else:
    labels = np.zeros([data.shape[0],1])
  return np.append(labels,data, axis=1)


def get_normalizers(data):
  global arrays_of_means
  global arrays_of_std
  arrays_of_means = np.empty([actors, 274])
  arrays_of_std = np.empty([actors, 274])
  for actor in range (0, actors):
    actordata = data[actor]
    actordata = actordata[:,1:]
    arrays_of_means[actor]=np.mean(actordata, axis = 0)
    arrays_of_std[actor]=np.std(actordata, axis = 0)

def normalize(data):
  global arrays_of_means
  global arrays_of_std
  res = data
  for actor in range (0, actors):
    actordata = data[actor]
    actordata = actordata[:,1:]
    actordata = np.subtract(actordata, arrays_of_means[actor])
    actordata = np.divide(actordata, arrays_of_std[actor])
    res[actor, :, 1:]=actordata
  return res


def getTrainingDataFB():
  data = np.empty([actors, 600, 275])
  for actor in range(0, actors):
    coverpath = '%(path)sdatasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"path":homePath, "fol":"fb4m-500-cover", "num":actor+1}
    coverdata = np.genfromtxt(coverpath, delimiter=' ')
    coverdata = coverdata[:300, :-1]

    stegpath = '%(path)sdatasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"path":homePath, "fol":"fb4m-500-stego-0.05", "num":actor+1}
    stegdata = np.genfromtxt(stegpath, delimiter=' ')
    stegdata = stegdata[:300, :-1]

    coverdata = addLabels(coverdata, False)
    stegdata = addLabels(stegdata, True)

    data[actor] = np.append(coverdata, stegdata, axis = 0)
  return data

def getTestingDataFB():
  data = np.empty([actors, 200, 275])
  for actor in range(0, actors):
    coverpath = '%(path)sdatasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"path":homePath, "fol":"fb4m-500-cover", "num":actor+1}
    coverdata = np.genfromtxt(coverpath, delimiter=' ')
    coverdata = coverdata[300:400, :-1]

    stegpath = '%(path)sdatasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"path":homePath, "fol":"fb4m-500-stego-0.05", "num":actor+1}
    stegdata = np.genfromtxt(stegpath, delimiter=' ')
    stegdata = stegdata[300:400, :-1]

    coverdata = addLabels(coverdata, False)
    stegdata = addLabels(stegdata, True)

    data[actor] = np.append(coverdata, stegdata, axis = 0)
  return data



def Train(D, r1, r2, iters, threshold):
  get_normalizers(D)
  normalize(D)
  w = np.zeros([D.shape[0]*(D.shape[2]-1)], 'f')
  print "Train: w"
  print w.shape
  print "Optimising with BFGS:"
  w = op.fmin_bfgs(function, w, args=(D, r1, r2), fprime=function_grad, \
                   callback=report, maxiter=iters, \
                   gtol=threshold)
    
  print 'Final cross entropy', multi_objective(w, D, r1, r2)
  w = np.reshape(w, (D.shape[0], w.shape[0]/D.shape[0]))
  return w

def TestSimple(data, w):
  N, K = data.shape
  D = 0 # the target classification is the second column
  K = K-D-1
  xtest = data[:,-K:]
  ttest = data[:,D]
  res = 0.0
  total = 0.0
  for i,log_p in enumerate(log_sigmoid(np.dot(xtest,w))):
    ires = np.exp(log_p)
    total = total+1
    if (ires < 0.5 and ttest[i]==0)or(ires>=0.5 and ttest[i]==1):
      res = res+1
  res = res/total
  print 'Accuracy: ', res
  return res

def Test(D, W):
  final_res = np.empty(W.shape[0])
  normalize(D)
  for i in range(0,D.shape[0]):
    res = TestSimple(D[i], W[i])
    print 'Actor %(num1)d result: %(num2)f' %{"num1":i, "num2":res}
    final_res[i] = res
  total = np.mean(final_res)
  print 'Total: %(num)f' %{"num":total}
  return final_res

def logResults(start, datashape, accuracies, end, comment, iters, threshold, regularisation1, regularisation2, payload):
  logPath = '%(path)slogs/log.txt' %{"path":homePath}
  fout = open(logPath, "a")
  fout.write("Simple Logistic Regression\n Start time: ")
  fout.write(start.strftime("%d-%m-%Y %H:%M:%S\n"))
  fout.write(" Data Size: ")
  fout.write('%(num1)04d x %(num2)04d' %{"num1":datashape[0], "num2":datashape[1]})
  fout.write("\n End Time: ")
  fout.write(end.strftime("%d-%m-%Y %H:%M:%S"))
  fout.write("\n Parameters:\n")
  fout.write('    iterations: %(num)d \n' %{"num":iters})
  fout.write('    regularisation1: %(num)f \n' %{"num":regularisation1})
  fout.write('    regularisation2: %(num)f \n' %{"num":regularisation2})
  fout.write('    threshold: %(num)e \n' %{"num":threshold})
  fout.write('    payload: %(num)f \n' %{"num":payload})
  fout.write('\nAccuracies:\n')
  for i in range (0, len(accuracies)):
    fout.write(' Actor %(num1)d: %(num2)f\n' %{"num1":i, "num2":accuracies[i]})
  fout.write("\nComment: \n")
  fout.write('%(num1)d Actors'%{"num1":actors})
  fout.write("\n----------------------------------------------\n")
  fout.close()

    
# This is the main function for training and testing the logistic 
# regression model.
def experiment(iters, threshold, regularisation1, regularisation2):
  comment = "Actors"
  start = datetime.datetime.now()
  data = getTrainingDataFB()
  myw = Train(data, regularisation1, regularisation2, iters, threshold)
  testData = getTestingDataFB()
  accuracies = Test(testData, myw)
  end = datetime.datetime.now()
  logResults(start, data.shape, accuracies, end, comment,iters, threshold, regularisation1, regularisation2, 0.05)

"""
Utility exception for generating a help message.
"""
class Usage(Exception):
  def __init__(self, msg):
    self.msg = msg


"""
This is an example main function for your experiment.
It demonstrates how to initialise and parse command
line arguments and then call the specific experiment
function.
"""
def main(argv=None):
  iters= 4000
  regularisation1 =1e1
  regularisation2 = 1e1
  threshold = 1e1

  # run our experiment function
  experiment(iters, threshold, regularisation1, regularisation2)



# This is how python handles main functions.
# __name__ won't equal __main__ be defined if 
# you import this file rather than run it on 
# the command line.
if __name__ == "__main__":
  sys.exit(main())

