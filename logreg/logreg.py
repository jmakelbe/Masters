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

# This function should calculate the negative conditional 
# log probability of the data x given weights w and 
# observed response variables y.
def objective(x, y, w):
  z = np.dot(x, w)
  # t = 1
  logC0 = -1*log_sigmoid(z)*y
  # t = 0
  logC1 = -1*log_sigmoid_complement(z)*(1-y)
  CrossEntropy = logC0 + logC1
  return CrossEntropy

# This is the log Gaussian prior 
def log_prior(w, alpha):
  return np.dot(w, w) / (2*alpha)

# This function should calculate the gradient of the negative 
# conditional log probability of the data x given weights w
# and observed response variables y.
def grad(x, y, w):
  mn = []
  for n in range(0, len(x) -1) :
    zn = np.dot(x[n], w)
    cn = np.exp(log_sigmoid(zn)) - y[n]
    mn.append(cn*x[n])
  Grad = sum(mn)
  return Grad

# This is the derivative of the Gaussian prior
def prior_grad(w, alpha):
  return w / alpha


