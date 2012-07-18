import numpy as np

W = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
W = np.reshape(W, (2, 5))

r1 = 1.0
r2 = 2.0
M = len(W)

meanW=np.add(np.square(r1), np.multiply(np.sum(W, axis=0), np.square(r1)))
meanW = np.divide(meanW, np.square(r2) + np.square(r1)*M)

res1 = np.multiply(meanW, 2*np.square(r1))
res1 = np.divide(res1, np.square(r1)*(np.square(r2)+M*(np.square(r1))))

res2 = np.multiply(np.subtract(W, meanW), np.square(r1))
res2 = np.sum(np.divide(res2, (np.square(r2)+M*np.square(r1))), axis=0)
res2 = np.multiply(np.divide(2, np.square(r2)), res2)

res = np.add(res1, res2)

print res
