#!/usr/bin/python

import numpy as np
import scipy.optimize as op
import sys, getopt
import datetime
from logreg import *
from util import *

actors = int(sys.argv[1])

# These three functions are helpers for the BFGS optimiser
last_func = 0
iterations = 0
def function(W, D, r1, r2): 
    global last_func
    last_func = sum(multi_objective(W, D, r1, r2)) 
    #last_func = last_func + regularisation(W, r1, r2)
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
        labels = np.ones([data.shape[0], 1])
    else:
        labels = np.zeros([data.shape[0], 1])
    return np.append(labels, data, axis=1)

def normalize(data):
    path = '/cluster/julie/code/normalizing/norms.csv'
    arrays = np.genfromtxt(path, delimiter=',')
    array_of_means = arrays[0]
    array_of_std = arrays[1]
    res = np.subtract(data, array_of_means)
    res = np.divide(res, array_of_std)
    return res

def getTrainingDataFB():
    data = np.empty([actors, 600, 275])
    for actor in range(0, actors):
        coverpath = '/cluster/julie/code/datasets/fb4m-500-cover/actor%(num)04d/data/merged.fea' % {"num":actor + 1}
        stegpath = '/cluster/julie/code/datasets/fb4m-500-stego-0.05/actor%(num)04d/data/merged.fea' % {"num":actor + 1}
        coverdata = np.genfromtxt(coverpath, delimiter=' ')
        stegdata = np.genfromtxt(stegpath, delimiter=' ')
        coverdata = normalize(coverdata[:300, :-1])
        stegdata = normalize(stegdata[:300, :-1])
	hundreds = np.ones(300)
	hundreds = np.multiply(hundreds, 1000)
        coverdata[:,0] = np.zeros(300)
        stegdata[:,0]=hundreds
        coverdata = addLabels(coverdata, False)
        stegdata = addLabels(stegdata, True)
        full = np.append(coverdata, stegdata, axis=0)
        data[actor] = full
    return data

def getTestingDataFB():
    data = np.empty([actors, 200, 275])
    for actor in range(0, actors):
        coverpath = '/cluster/julie/code/datasets/fb4m-500-cover/actor%(num)04d/data/merged.fea' % {"num":actor + 1}
        stegpath = '/cluster/julie/code/datasets/fb4m-500-stego-0.05/actor%(num)04d/data/merged.fea' % {"num":actor + 1}
        coverdata = np.genfromtxt(coverpath, delimiter=' ')
        stegdata = np.genfromtxt(stegpath, delimiter=' ')
        coverdata = normalize(coverdata[300:400, :-1])
        stegdata = normalize(stegdata[300:400, :-1])
        hundreds = np.ones(100)
        hundreds = np.multiply(hundreds, 1000)
        coverdata[:,0] = np.zeros(100)
        stegdata[:,0]=hundreds
	coverdata = addLabels(coverdata, False)
        stegdata = addLabels(stegdata, True)
        full = np.append(coverdata, stegdata, axis=0)
        data[actor] = full
    return data

def Train(D, r1, r2, iters, threshold):
    w = np.zeros([D.shape[0]*(D.shape[2]-1)], 'f')
    print "Train: w"
    print w.shape
    print "Optimising with BFGS:"
    w = op.fmin_bfgs(function, w, args=(D, r1, r2), fprime=function_grad, \
                     callback=report, maxiter=iters, \
                     gtol=threshold)
    
    print 'Final cross entropy', sum(multi_objective(w, D, r1, r2))
    w = np.reshape(w, (D.shape[0], w.shape[0]/D.shape[0]))
    print w
    return w

def TestSimple(data, w):
  N, K = data.shape
  D = 0 # the target classification is the second column
  K = K-D-1
  xtest = data[:,-K:]
  xtest = normalize(xtest)
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
    for i in range(0,D.shape[0]):
        res = TestSimple(D[i], W[i])
        print 'Actor %(num1)d result: %(num2)f' %{"num1":i, "num2":res}
        final_res[i] = res
    total = np.mean(final_res)
    print 'Total: %(num)f' %{"num":total}
    return final_res

def logResults(start, datashape, accuracies, end, comment, iters, threshold, regularisation1, regularisation2, payload):
    logPath = "/cluster/julie/code/logs/log.txt"
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

def experiment(iters, threshold, regularisation1, regularisation2):
    comment = "Actors"
    start = datetime.datetime.now()
    data = getTrainingDataFB()
    myw = Train(data, regularisation1, regularisation2, iters, threshold)
    accuracies = Test(getTestingDataFB(), myw)
    end = datetime.datetime.now()
    logResults(start, data.shape, accuracies, end, comment,iters, threshold, regularisation1, regularisation2, 0.05)


experiment(2000, 1e1, 1, -20)
