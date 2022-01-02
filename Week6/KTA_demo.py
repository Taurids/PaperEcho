# coding=utf-8
import numpy as np
from Echo7.kta import KTA
np.random.seed(0)
# input X, y
X = np.random.random((4, 2))
y = np.array([-1, -1, 1, 1])

# get Original Alignment
ori_A, opt_A = KTA(X, y, kernel='rbf', gamma=0.7)
print('Original Alignment: ', ori_A)
print('Optimal Alignment: ', opt_A)
