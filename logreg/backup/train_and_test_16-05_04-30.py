#!/usr/bin/python
"""
usage: python experiment1.py [option]
Options and arguments:
  -h, --help                         : print this help message
  -i, --iterations=arg               : max number of training iterations
  -n, --eta=arg                      : eta step size parameter in 
                                       gradient descent
  -t, --threshold=arg                : gradient tolerance termination 
  -r, --regularisation-parameter=arg : Gaussian sigma^2 parameter in the 
                                       logistic regression MAP objective
"""
import sys, getopt
import numpy as np
import scipy.optimize as op

from util import *
from logreg import *
import datetime


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

def concat(folder, test):
  full = np.empty([0, 274])
  for actor in range(1, 5):
    path = '/cluster/julie/code/datasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"fol":folder, "num":actor}
    data = np.genfromtxt(path, delimiter=' ')
    if test:
      data = data[401:, :]
    else:
      data = data[:400, :]
    full = np.append(full, data[:,:-1], axis=0)
  return full

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

# This is the main function for training and testing the logistic 
# regression model.
def experiment(iterations, regularisation, eta, threshold, testout):
  comment = raw_input("What is the comment for this experiment?\n")
  now = datetime.datetime.now()
  logPath = "/cluster/julie/code/logs/log.txt"
  fout = open(logPath, "a")
  fout.write("Simple Logistic Regression\n Start time: ")
  fout.write(now.strftime("%d-%m-%Y %H:%M:%S\n"))
  # read the data
  coverdata = concat("fb4m-500-cover", False)
  test_coverdata= concat("fb4m-500-cover", True)
  stegdata = concat("fb4m-500-stego-0.5", False)
  test_stegdata = concat("fb4m-500-stego-0.5", True)

  coverdata = addLabels(coverdata, False)
  stegdata = addLabels(stegdata, True)
  test_coverdata = addLabels(test_coverdata, False)
  test_stegdata = addLabels(test_stegdata, True)

  data = np.append(coverdata, stegdata, axis=0)
  test_data = np.append(test_coverdata, test_stegdata, axis=0)

  fout.write(" Data Size: ")
  fout.write('%(num1)04d x %(num2)04d' %{"num1":data.shape[0], "num2":data.shape[1]})
  fout.write("\n Test Data Size: ")
  fout.write('%(num1)04d x %(num2)04d' %{"num1":test_data.shape[0], "num2":test_data.shape[1]})
  fout.write("\n Payload: 0.5")

  # the last K columns are the features
  x = data[:,1:]
  # the second column is the response variable
  t = data[:,0]
  xtest = test_data[:,1:]
  ttest = test_data[:,0]
  x = normalize(x)
  w = np.zeros(x.shape[1], 'f')


  print "Optimising with Gradient Descent:"
  for i in range(iterations):
    gradient_vector=function_grad(w, x, t, regularisation)
    w-= eta * gradient_vector 
    if not (i% 10):
      print '   iteration', i, ': cross entropy=', function(w, x, t, regularisation)

  final = sum(objective(x, t, w))
  print 'Final cross entropy', final
  fout.write("\n Final Cross Entropy: ")
  fout.write('%(num)04d' %{"num":final})
  # label the test data using the learnt feature weights w
  out = open(testout,'w')
  for i,log_p in enumerate(log_sigmoid(np.dot(xtest, w))):
    print >>out,'%d,%f' % (i+1,np.exp(log_p))
  out.close()
  
  now = datetime.datetime.now()
  fout.write("\n End Time: ")
  fout.write(now.strftime("%d-%m-%Y %H:%M:%S"))
  fout.write("\n Parameters:\n")
  fout.write('    iterations: %(num)d \n' %{"num":iterations})
  fout.write('    eta: %(num)d \n' %{"num":eta})
  fout.write('    regularisation: %(num)d \n' %{"num":regularisation})
  fout.write('    threshold: %(num)d \n' %{"num":threshold})
  fout.write("\nComment: \n")
  fout.write(comment)
  fout.write("\n----------------------------------------------\n")  
  fout.close()
"""
Utility exception for generating a help message.
"""
class Usage(Exception):
  def __init__(self, msg):
    self.msg = msg


"""
Demonstrates how to initialise and parse command
line arguments and then call the specific experiment
function.
"""
def main(argv=None):
  experiment_args = {}
  experiment_args["iterations"] =2000
  experiment_args["eta"] = 1e-6
  experiment_args["regularisation"] = 1e-3
  experiment_args["threshold"] = 1e1
  experiment_args["test-out"] = "/cluster/julie/code/logreg/logreg-entry.csv"

  # Example command line argument processing.
  # You can add any arguments you see fit. 
  if argv is None:
    argv = sys.argv
  try:
    try:
      opts, args = getopt.getopt(argv[1:], "hi:n:r:t:o:", 
                                 ["help",
                                  "iterations=",
                                  "eta=",
                                  "threshold=",
                                  "test-out",
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
  experiment(experiment_args["iterations"], experiment_args["regularisation"], experiment_args["eta"], experiment_args["threshold"], experiment_args["test-out"])


# This is how python handles main functions.
# __name__ won't equal __main__ be defined if 
# you import this file rather than run it on 
# the command line.
if __name__ == "__main__":
  sys.exit(main())
