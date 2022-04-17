from Echo7 import stats
import numpy as np

e = np.array(1e-20)  # epsilon
X = np.linspace(-0.5, 0.5, num=11, dtype=float)
Y = np.linspace(-0.2, 0.2, num=5, dtype=float)
pxy = stats.norm2_pdf(X, Y)
px = np.sum(pxy, axis=1)
py = np.sum(pxy, axis=0)
qx = stats.norm_pdf(X)

# Entropy
HX = -np.sum(px * np.log(px+e))
print('Entropy\'s Value:', HX)

# Joint Entropy
HXY = -np.sum(pxy * np.log(pxy+e))
print('Joint Entropy\'s Value:', HXY)

# Conditional Entropy
p = pxy / np.sum(pxy, axis=1, keepdims=True)
HY_X = -np.sum(p * np.log(p+e), axis=1)
HY_X = np.sum(px * HY_X)
print('Conditional Entropy\'s Value:', HY_X)

# Relative Entropy
Dpq = np.sum(px * np.log(px/qx+e))
print('Relative Entropy\'s Value:', Dpq)

# Mutual Information
IXY = 0
for i in range(len(px)):
    for j in range(len(py)):
        IXY += pxy[i][j] * np.log(pxy[i][j] / (px[i] * py[j] + e) + e)
print('Mutual Information\'s Value:', IXY)
