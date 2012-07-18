#!/usr/bin/python

import sys, getopt
import numpy as np
import scipy.optimize as op
from logreg import *

last_func=0
iterations=0

def function(w, x, t, r):
  global last_func
  last_func = sum(objective(x,t,w)) + log_prior(w, r)
  return last_func

def function_grad(w, x, t, r):
  return grad(x,t,w)+prior_grad(w, r)

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

def normalize(data):
  means = np.mean(data, axis = 0)
  var = np.std(data, axis = 0)
  res = np.subtract(data, means)
  res = np.divide(res, var)
  return res

def getdata():
  path = '/cluster/julie/code/datasets/20000jpg_cover/data/merged.fea'
  stegpath = '/cluster/julie/code/datasets/20000jpg_stego_0.5/data/merged.fea'
  coverdata = np.genfromtxt(path, delimiter=' ')
  stegdata = np.genfromtxt(stegpath, delimiter=' ')

  coverdata = coverdata[:5000, :-1]
  stegdata = stegdata[:5000, :-1]
  coverdata = addLabels(coverdata, False)
  stegdata = addLabels(stegdata, True)

  data = np.append(coverdata, stegdata, axis=0)

  return data

def getTestData():
  path = '/cluster/julie/code/datasets/20000jpg_cover/data/merged.fea'
  stegpath = '/cluster/julie/code/datasets/20000jpg_stego_0.5/data/merged.fea'
  coverdata = np.genfromtxt(path, delimiter=' ')
  stegdata = np.genfromtxt(stegpath, delimiter=' ')

  coverdata = coverdata[2501:3000, :-1]
  stegdata = stegdata[2501:3000, :-1]

  data = np.append(coverdata, stegdata, axis=0)

  return data


def Train(cmdline_args, data):
  # the last K columns are the features
  x = data[:,1:]
  # the second column is the response variable
  t = data[:,0]
  x = normalize(x)

  w = np.zeros(x.shape[1], 'f')

  print "Optimising with BFGS:"
  w = op.fmin_bfgs(function, w, args=(x,t,cmdline_args["regularisation"]), fprime=function_grad, \
                   callback=report, maxiter=cmdline_args["iterations"], \
                   gtol=cmdline_args["threshold"])


  print 'Final cross entropy', sum(objective(x, t, w))
  return w

def Test(cmdline_args, testset, w):
  # label the test data using the learnt feature weights w
  out = open(cmdline_args['test-out'],'w')
  for i,log_p in enumerate(log_sigmoid(np.dot(testset,w))):
    print >>out,'%d,%f' % (i+1,np.exp(log_p))
  out.close()


def experiment(cmdline_args):
  data = getdata()
  w = Train(cmdline_args, data)
  testdata = getTestData()
  Test(cmdline_args, testdata, w)
  
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
  experiment_args = {}
  experiment_args["iterations"] = 1000
  experiment_args["eta"] = 1e-11
  experiment_args["regularisation"] = 1e-3
  experiment_args["threshold"] = 1e1
  experiment_args["test-out"] = "logreg-entry.csv"

  # Example command line argument processing.
  # You can add any arguments you see fit. 
  if argv is None:
    argv = sys.argv
  try:
    try:
      opts, args = getopt.getopt(argv[1:], "hi:n:r:t:bo:",
                                 ["help",
                                  "iterations=",
                                  "eta=",
                                  "threshold=",
                                  "test-out=",
                                  "regularisation-parameter="])
    except getopt.error, msg:
      raise Usage(msg)

    # process options
    for option, argument in opts:
      if option in ("-h", "--help"):
        print __doc__
        sys.exit(0)
      if option in ("-i", "--iterations"):
        experiment_args["iterations"] =int(argument)
      if option in ("-n", "--eta"):
        experiment_args["eta"] = float(argument)
      if option in ("-t", "--threshold"):
        experiment_args["threshold"] = float(argument)
      if option in ("-r", "--regularisation-parameter"):
        experiment_args["regularisation"] = float(argument)
      if option in ("-o", "--test-out"):
        experiment_args["test-out"] = argument

  except Usage, err:
    print "Error parsing command line arguments:"
    print >>sys.stderr, "  %s" % err.msg
    print __doc__
    return 2

  # run our experiment function
  experiment(experiment_args)


# This is how python handles main functions.
# __name__ won't equal __main__ be defined if 
# you import this file rather than run it on 
# the command line.
if __name__ == "__main__":
  sys.exit(main())

