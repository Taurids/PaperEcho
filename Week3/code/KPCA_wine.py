# coding=utf-8
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt

my_parser = argparse.ArgumentParser()
# Required parameters
my_parser.add_argument("--n_components", default=2, type=int, help="")
my_parser.add_argument("--kernel", default='rbf', type=str, help="Select the kernel function")
my_parser.add_argument("--need_svd", action='store_true', help="Whether using the direct svd function.")
my_parser.add_argument("--pre_mean", action='store_true', help="Calculate the mean of each column.")
my_parser.add_argument("--pre_std", action='store_true', help="Calculate the std of each column.")

args = my_parser.parse_args()

# ------------------------------ Data Preprocess  -----------------------------------
path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
df = pd.read_csv(path, header=None)
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
# StandardScaler  Q：为什么需要去均值和方差归一化？
x_means = np.array(np.mean(np.mat(X), axis=0)) if args.pre_mean else 0
x_stds = np.array(np.std(np.mat(X), axis=0)) if args.pre_std else 1
X = (X - x_means) / x_stds

# -------------------------------- PCA method --------------------------------------
# first method -> svd
if args.need_svd:
    U, sigma, VT = np.linalg.svd(X.T)
    x_pca = np.matmul(X, U[:, :args.n_components])
# second method -> paper solver
else:
    C = np.matmul(X.T, X) / X.shape[0]
    eigenvalue, eigenvector = np.linalg.eigh(C)
    eigenvector = eigenvector[:, ::-1][:, :args.n_components]
    x_pca = np.matmul(X, eigenvector)

# ----------------------------- Data Visualization ---------------------------------
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax[0].scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x_pca[y == 2, 0], x_pca[y == 2, 1], color='blue', marker='o', alpha=0.5)
ax[0].scatter(x_pca[y == 3, 0], x_pca[y == 3, 1], color='lightgreen', marker='s', alpha=0.5)
ax[1].scatter(x_pca[y == 1, 0], np.zeros((len(x_pca[y == 1, 0]), 1)) + 0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(x_pca[y == 2, 0], np.zeros((len(x_pca[y == 2, 0]), 1)) + 0.02, color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_pca[y == 3, 0], np.zeros((len(x_pca[y == 3, 0]), 1)) + 0.02, color='lightgreen', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# -------------------------------- KPCA method --------------------------------------

#  select kernel function and make K
if args.kernel == 'rbf':
    sigma = 1
    mat = np.sum(X ** 2, 1).reshape(-1, 1) + np.sum(X ** 2, 1) - 2 * np.dot(X, X.T)
    K = np.exp(-0.5 / sigma ** 2 * mat)
elif args.kernel == 'tanh':
    theta = 1
    K = np.tanh(np.matmul(X, X.T) + theta)
elif args.kernel == 'poly':
    theta, p = 1, 3
    K = np.power(np.matmul(X, X.T) + theta, p)
else:  # 'linear'
    K = np.matmul(X, X.T)

# Centralize the kernel matrix
N = K.shape[0]
one_n = np.ones((N, N)) / N
K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

eigenvalue, eigenvector = np.linalg.eigh(K)  # 关注一下与np.linalg.eig的区别
x_kpca = eigenvector[:, ::-1][:, :args.n_components]


# ----------------------------- Data Visualization ---------------------------------
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

ax[0].scatter(x_kpca[y == 1, 0], x_kpca[y == 1, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x_kpca[y == 2, 0], x_kpca[y == 2, 1], color='blue', marker='o', alpha=0.5)
ax[0].scatter(x_kpca[y == 3, 0], x_kpca[y == 3, 1], color='lightgreen', marker='s', alpha=0.5)
ax[1].scatter(x_kpca[y == 1, 0], np.zeros((len(x_kpca[y == 1, 0]), 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(x_kpca[y == 2, 0], np.zeros((len(x_kpca[y == 2, 0]), 1))+0.02, color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_kpca[y == 3, 0], np.zeros((len(x_kpca[y == 3, 0]), 1))+0.02, color='lightgreen', marker='s', alpha=0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()
