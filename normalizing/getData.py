#!/usr/bin/python

import numpy as np

def getDataFB():
# read the data
  coverdata = concat("fb4m-500-cover", False)
  stegdata = concat("fb4m-500-stego-0.05", False)
  data = np.append(coverdata, stegdata, axis=0)
  return data

def concat(folder, test):
  full = np.empty([0, 274])
  for actor in range(1,5):
    path = '/cluster/julie/code/datasets/%(fol)s/actor%(num)04d/data/merged.fea' %{"fol":folder, "num":actor}
    data = np.genfromtxt(path, delimiter=' ')
    full = np.append(full, data[:,:-1], axis=0)
  return full

def get_normalizers(data):
  global array_of_means
  global array_of_std
  array_of_means = np.mean(data, axis = 0)
  array_of_std = np.std(data, axis = 0)
  data = np.empty([2, 274])
  data[0] = array_of_means
  data[1] = array_of_std
  path = '/cluster/julie/code/normalizing/norms.csv'
  np.savetxt(path, data, delimiter=",")
  print "done"

get_normalizers(getDataFB())
