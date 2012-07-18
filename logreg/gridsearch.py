#!/usr/bin/python

from train_and_test import *
import numpy as np
import sys

j = int(sys.argv[1])

iters= 4000
eta = 1e-11
regularisation = 10
threshold = 1e-11
bfgs = True 
cross = 10
payload = 0.05

logPath = "/cluster/julie/code/logs/simpleGrid.txt"
#fout = open(logPath, "a")
#fout.write("Simple Logistic Regression Gridsearch\n\n")
#fout.write("linesearch: regularisation\n\n")
#fout.close()

fout = open(logPath, "a")
regularisation = np.exp(j)
accuracy = experiment(iters, eta, threshold, regularisation, cross, bfgs, payload)
fout.write('regularisation %(num1)03d : accuracy %(num2)f ' %{"num1": j, "num2":accuracy})
fout.write('\n')
fout.close()

