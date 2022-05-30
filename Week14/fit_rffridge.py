"""============================================================================
Fitting and plotting script for kernel ridge regression (see rffridge.py).
For more, see the accompanying blog post:
http://gregorygundersen.com/blog/2019/12/23/random-fourier-features/
============================================================================"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from rffridge import RFFRidgeRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

N = 2000
X = np.linspace(-10, 10, N)[:, None]
mean = np.zeros(N)
cov = RBF()(X.reshape(N, -1))
y = np.random.multivariate_normal(mean, cov)
noise = np.random.normal(0, 0.5, N)
y += noise

# Finer resolution for smoother curve visualization.
X_test = np.linspace(-10, 10, N*2)[:, None]

# Set up figure and plot data.
fig, axes = plt.subplots(3, 1)
fig.set_size_inches(10, 5)
ax1, ax2, ax3 = axes
cmap = plt.cm.get_cmap('Blues')

ax1.scatter(X, y, s=30, c=[cmap(0.3)])
ax2.scatter(X, y, s=30, c=[cmap(0.3)])
ax3.scatter(X, y, s=30, c=[cmap(0.3)])

# Fit ridge regression using Ridge.
start_time = time.time()
clf = Ridge(alpha=500)
clf = clf.fit(X, y)
half_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
ax1.plot(X_test, y_pred, c=cmap(0.9))
print('训练时间：', half_time - start_time)
print('测试时间：', end_time - half_time)
print('MSE为：', mean_squared_error(X_test, y_pred))
print('r2_score为：', r2_score(X_test, y_pred))


# Fit kernel ridge regression using an RBF kernel.
start_time = time.time()
clf = KernelRidge(kernel=RBF())
clf = clf.fit(X, y)
half_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
ax2.plot(X_test, y_pred, c=cmap(0.9))
print('训练时间：', half_time - start_time)
print('测试时间：', end_time - half_time)
print('MSE为：', mean_squared_error(X_test, y_pred))
print('r2_score为：', r2_score(X_test, y_pred))

# Fit kernel ridge regression using random Fourier features.
start_time = time.time()
rff_dim = 1000
clf = RFFRidgeRegression(rff_dim=rff_dim)
clf.fit(X, y)
half_time = time.time()
y_pred = clf.predict(X_test)
end_time = time.time()
ax3.plot(X_test, y_pred, c=cmap(0.9))
print('训练时间：', half_time - start_time)
print('测试时间：', end_time - half_time)
print('MSE为：', mean_squared_error(X_test, y_pred))
print('r2_score为：', r2_score(X_test, y_pred))

# Labels, etc.
ax1.margins(0, 0.1)
ax1.set_title('Ridge regression')
ax1.set_ylabel(r'$y$', fontsize=14)
ax1.set_xticks([])
ax2.margins(0, 0.1)
ax2.set_title('RBF kernel regression')
ax2.set_ylabel(r'$y$', fontsize=14)
ax2.set_xticks([])
ax3.margins(0, 0.1)
ax3.set_title(rf'RFF ridge regression, $R = {rff_dim}$')
ax3.set_ylabel(r'$y$', fontsize=14)
ax3.set_xlabel(r'$x$', fontsize=14)
ax3.set_xticks([])
# ax3.set_xticks(np.arange(-10, 10.1, 1))
plt.tight_layout()
plt.show()
