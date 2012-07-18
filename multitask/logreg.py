import sys
import numpy as np
import scipy.optimize as op
import util

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_sigmoid(z):
    # n.b. -np.logaddexp(0,-z) calculates -log(1+exp(-z)) 
    # in without exponentiation to avoid overflow.
    # Try help(numpy.logaddexp) in the interpreter for
    # more information.
    return -np.logaddexp(0,-z)

def log_sigmoid_complement(z):
    return -np.logaddexp(0,z)

# This function should calcul1ate the negative conditional 
# log probability of the data x given weights w and 
# observed response variables y.
def simple_objective(x, y, w):
    z = np.dot(x, w)
    # t = 1
    logC0 = -1*log_sigmoid(z)*y
    # t = 0
    logC1 = -1*log_sigmoid_complement(z)*(1-y)
    CrossEntropy = logC0 + logC1
    return CrossEntropy


def simple_grad(x, y, w):
    mn = []
    for n in range(0, len(x) -1) :
        zn = np.dot(x[n], w)
        cn = np.exp(log_sigmoid(zn)) - y[n]
        mn.append(cn*x[n])
    Grad = sum(mn)
    return Grad


def likelihood(W,D):
    final_entropy = np.empty([D.shape[0], D.shape[1]])
    for i in range(0,D.shape[0]):
        actordata = D[i, :, :]
        N,K = actordata.shape
        K = K-1
        x = actordata[:,-K:]
        y = actordata[:,0]
        smallw = W[i]
        final_entropy[i]= simple_objective(x, y, smallw)
    return final_entropy

def grad_likelihood(W, D):
    final_grad = np.empty([W.shape[0], W.shape[1]])
    for i in range(0,D.shape[0]):
        actordata = D[i, :, :]
        N,K = actordata.shape
        K = K-1
        x = actordata[:,-K:]
        y = actordata[:,0]
        smallw = W[i]
        final_grad[i]= simple_grad(x, y, smallw)
    return final_grad


def regularisation(W, r1, r2):
    M = len(W)
    meanW=np.add(np.square(r1), np.multiply(np.sum(W, axis=0), np.square(r1)))
    meanW = np.divide(meanW, np.square(r2) + np.square(r1)*M)
    res1=np.divide(np.square(meanW), np.square(r1))
    res2=np.divide(np.sum(np.square(np.subtract(W, meanW)), axis = 0), np.square(r2))
    res = np.add(res1, res2)
    res = np.sum(res)
    return res

def grad_regularisation(W, r1, r2):   
    M = len(W)

    meanW=np.add(np.square(r1), np.multiply(np.sum(W, axis=0), np.square(r1)))
    meanW = np.divide(meanW, np.square(r2) + np.square(r1)*M)

    res1 = np.multiply(meanW, 2*np.square(r1))
    res1 = np.divide(res1, np.square(r1)*(np.square(r2)+M*(np.square(r1))))
    
    res2 = np.multiply(np.subtract(W, meanW), np.square(r1))
    res2 = np.sum(np.divide(res2, (np.square(r2)+M*np.square(r1))), axis=0)
    res2 = np.multiply(np.divide(2, np.square(r2)), res2)
    
    res = np.add(res1, res2)

    return res

def multi_objective(W, D, r1, r2):
    W = np.reshape(W, (D.shape[0], W.shape[0]/D.shape[0]))
    res = likelihood(W, D)
    res = np.reshape(res, (res.shape[0]*res.shape[1]))
    return res


def grad_multi_objective(W, D, r1, r2):
    W = np.reshape(W, (D.shape[0], W.shape[0]/D.shape[0]))
    res1 = grad_likelihood(W, D)
    res2 = grad_regularisation(W, r1, r2)
    res = res1
    res = np.reshape(res, (res.shape[0]*res.shape[1]))
    return res

