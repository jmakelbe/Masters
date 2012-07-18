import sys, numpy, inspect


def todo():
  print "You should implement this function: %s" % inspect.stack()[1][3]
  sys.exit(1)


"""
Split data into 'splits' different sub-blocks, returning the
block indexed by 'test_split' plus the rest of the data.
"""
def split_data(splits, test_split, data):
  assert test_split >= 0 and test_split < splits
  assert splits <= len(data)

  split_size = len(data) / int(splits)
  test_start = split_size*test_split
  test_end   = test_start + split_size

  test  = data[test_start:test_end]
  train = numpy.concatenate([data[0:test_start],data[test_end:]])

  return (train,test)
