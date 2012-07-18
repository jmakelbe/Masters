#!/usr/bin/python

import sys, getopt
import numpy as np
import scipy.optimize as op
import datetime

from logreg import *
from util import *
#logreg
homePath = '/Users/July/Documents/Dropbox/OxfordMSc/Thesis/code/'
#homePath = '/cluster/julie/code/'

actors = 1

crossvalid = True
regular = False

# These three functions are helpers for the BFGS optimiser
last_func=0
iterations=0
array_of_means=0
array_of_std=0
def function(w, x, t, r): 
  global last_func
  last_func = sum(objective(x,t,w))
  if regular:
    last_func = last_func + log_prior(w, r)
  return last_func

def function_grad(w, x, t, r):
  res = grad(x, t, w)
  if regular:
    res = res + prior_grad(w, r)
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
  global array_of_means
  global array_of_std
  array_of_means = np.mean(data, axis = 0)
  array_of_std = np.std(data, axis = 0)

def normalize(data):
  global array_of_means
  global array_of_std
  res = np.subtract(data, array_of_means)
  res = np.divide(res, array_of_std)
  return res

def getTrainingData20k():
  path = '/cluster/julie/code/datasets/20000jpg_cover/data/merged.fea'
  stegpath = '/cluster/julie/code/datasets/20000jpg_stego_0.5/data/merged.fea'
  coverdata = np.genfromtxt(path, delimiter=' ')
  stegdata = np.genfromtxt(stegpath, delimiter=' ')

  coverdata = coverdata[:2500, :-1]
  stegdata = stegdata[:2500, :-1]
  coverdata = addLabels(coverdata, False)
  stegdata = addLabels(stegdata, True)
  
  data = np.append(coverdata, stegdata, axis=0)
 
  return data


def getTestData20k():
  path = '/cluster/julie/code/datasets/20000jpg_cover/data/merged.fea'
  stegpath = '/cluster/julie/code/datasets/20000jpg_stego_0.5/data/merged.fea'
  coverdata = np.genfromtxt(path, delimiter=' ')
  stegdata = np.genfromtxt(stegpath, delimiter=' ')

  coverdata = coverdata[2501:3000, :-1]
  stegdata = stegdata[2501:3000, :-1]
  coverdata = addLabels(coverdata, False)
  stegdata = addLabels(stegdata, True)

  data = np.append(coverdata, stegdata, axis=0)

  return data

def getTrainingDataFB():
# read the data
  coverdata = concat("fb4m-500-cover", False)
  stegdata = concat("fb4m-500-stego-0.05", False)
  #hundreds = np.ones(400)
  #hundreds = np.multiply(hundreds, 100)
  #coverdata[:,0] = np.zeros(400)
  #stegdata[:,0]=hundreds
  coverdata = addLabels(coverdata, False)
  stegdata = addLabels(stegdata, True)
  data = np.append(coverdata, stegdata, axis=0)

  return data

def getTestDataFB():
  test_coverdata= concat("fb4m-500-cover", True)
  test_stegdata = concat("fb4m-500-stego-0.05", True)
  #hundreds = np.ones(400)
  #hundreds = np.multiply(hundreds, 100)
  #coverdata[:,0] = np.zeros(400)
  #stegdata[:,0]=hundreds
  test_coverdata = addLabels(test_coverdata, False)
  test_stegdata = addLabels(test_stegdata, True)

  test_data = np.append(test_coverdata, test_stegdata, axis=0)
  return test_data


def concat(folder, test):
  full = np.empty([0, 274])
  for actor in range(1,2):
    path = '%(path)sdatasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"path":homePath, "fol":folder, "num":actor}
    data = np.genfromtxt(path, delimiter=' ')
    if test:
      data = data[400:, :]
    else:
      data = data[:400, :]
    full = np.append(full, data[:,:-1], axis=0)
  return full



def Train(data, iters, eta, threshold, regularisation, cross, bfgs, payload):
  N, K = data.shape
  D = 0 # the target classification is the second column
  K = K-D-1

  # the last K columns are the features
  x = data[:,-K:]
  get_normalizers(x)
  x = normalize(x)
  # the second column is the response variable
  t = data[:,D]
  
  w = np.zeros(x.shape[1], 'f')

  if bfgs:
    # Use the quasi-Newton convex optimisation algorithm BFGS from Scipy. It's much more 
    # effective than gradient descent.
    # BFGS is an iterative algorithm which calls function(w) and func_grad(w) on each iteration to
    # calculate the objective function and its gradient. The function report(w) is also called
    # to enable the progress of the optimisation to be reported to stdout. gtol is the gradient
    # tolerance, if the norm of the gradient doesn't change by at least this much between 
    # iterations the optimiser will terminate.
    print "Optimising with BFGS:"
    w = op.fmin_bfgs(function, w, args=(x,t,regularisation), fprime=function_grad, \
                     callback=report, maxiter=iters, \
                     gtol=threshold)
  else: # otherwise use gradient descent
    print "Optimising with Gradient Descent:"
    for i in range(iters):
      w -= eta * function_grad(w, x, t, regularisation)
      if not (i% 10):
        print '   iteration', i, ': cross entropy=', function(w, x, t, regularisation)

  print 'Final cross entropy', sum(objective(x, t, w))
  return w

def Test(data, w):
  N, K = data.shape
  D = 0 # the target classification is the second column
  K = K-D-1
  xtest = data[:,-K:]
  xtest = normalize(xtest)
  print xtest
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

def crossvalidate(iters, eta, threshold, regularisation, cross, bfgs, payload, data):
  accuracy=0.0
  number = cross
  for i in range (0, number):
    global iterations
    global last_func
    iterations = 0
    last_func = 0
    split = split_data(number, i, data)
    train_data = split[0]
    test_data = split[1]
    w = Train(train_data, iters, eta, threshold, regularisation, cross, bfgs, payload)
    accuracy = accuracy + Test(test_data, w)
  accuracy = accuracy/number
  print ("Total accuracy: %(num)f" %{"num":accuracy})
  return accuracy

def logResults(start, datashape, accuracy, end, comment, iters, eta, threshold, regularisation, cross, bfgs, payload):
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
  fout.write('    eta: %(num)e \n' %{"num":eta})
  fout.write('    regularisation: %(num)f \n' %{"num":regularisation})
  fout.write('    threshold: %(num)e \n' %{"num":threshold})
  fout.write('    bfgs: %(num)d \n' %{"num":bfgs})
  fout.write('    cross: %(num)d \n' %{"num":cross})
  fout.write('    payload: %(num)f \n' %{"num":payload})
  fout.write('\nAccuracy: %(num)f \n' %{"num":accuracy})
  fout.write("\nComment: \n")
  fout.write(comment)
  fout.write("\n----------------------------------------------\n")
  fout.close()

# This is the main function for training and testing the logistic 
# regression model.
def experiment(iters, eta, threshold, regularisation, cross, bfgs, payload):
  #comment = raw_input("What is the comment for this experiment?\n")
  comment = "5 Actors"
  start = datetime.datetime.now()
  accuracy=0.0
  if crossvalid:
    data = getTrainingDataFB()
    np.random.shuffle(data)
    np.random.shuffle(data)
    np.random.shuffle(data)
    np.random.shuffle(data)
    np.random.shuffle(data)
    accuracy+=crossvalidate(iters, eta, threshold, regularisation, cross, bfgs, payload, data)    
  else:
    data = getOncActorTrainData()
    testdata = getTestDataFB()
    w = Train(data, iters, eta, threshold, regularisation, cross, bfgs, payload)
    accuracy = accuracy + Test(testdata, w)
  end = datetime.datetime.now()
  logResults(start, data.shape, accuracy, end, comment, iters, eta, threshold, regularisation, cross, bfgs, payload)
  return accuracy

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
  eta = 1e-11
  regularisation =1e20 
  threshold = 1e1
  bfgs = True
  cross = 4
  payload = 0.05

  # Example command line argument processing.
  # You can add any arguments you see fit. 
  if argv is None:
    argv = sys.argv
  try:
    try:
      opts, args = getopt.getopt(argv[1:], "hi:n:r:p:t:bc:", 
                                 ["help",
                                  "iterations=",
                                  "eta=",
                                  "threshold=",
                                  "cross=",
                                  "bfgs",
                                  "payload=",
                                  "regularisation-parameter="])
    except getopt.error, msg:
      raise Usage(msg)

    # process options
    for option, argument in opts:
      if option in ("-h", "--help"):
        print __doc__
        sys.exit(0)
      if option in ("-i", "--iterations"):
        iters =int(argument)
      if option in ("-n", "--eta"):
        eta = float(argument)
      if option in ("-t", "--threshold"):
        threshold = float(argument)
      if option in ("-r", "--regularisation-parameter"):
        regularisation = float(argument)
      if option in ("-c", "--cross"):
        cross = int(argument)
      if option in ("-b", "--bfgs"):
        bfgs = True
      if option in ("-p", "--payload"):
        payload = float(argument)

  except Usage, err:
    print "Error parsing command line arguments:"
    print >>sys.stderr, "  %s" % err.msg
    print __doc__
    return 2
  
  # run our experiment function
  experiment(iters, eta, threshold, regularisation, cross, bfgs, payload)


# This is how python handles main functions.
# __name__ won't equal __main__ be defined if 
# you import this file rather than run it on 
# the command line.
if __name__ == "__main__":
  sys.exit(main())

