import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# 数据预处理
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data', header=None)
x, y = df.iloc[:, 1:-1].values, df.iloc[:, -1].values
sc = StandardScaler()
x = sc.fit_transform(x)

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
# 可视化
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
colors = ['red', 'blue', 'green', 'cyan', 'yellow', 'magenta', 'black']
markers = ['^', 'o', 'v', '<', '>', '8', 's', 'p', '*', '+', 'h', 'H']
for i in range(1, 8):
    ax[0].scatter(
        x_pca[y == i, 0], x_pca[y == i, 1],
        color=colors[i-1], marker=markers[i-1], alpha=0.5
    )
    ax[1].scatter(
        x_pca[y == i, 0], np.zeros((len(x_pca[y == i, 0]), 1)) + 0.02,
        color=colors[i-1], marker=markers[i-1], alpha=0.5
    )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()


# KPCA
kpca = KernelPCA(n_components=2, kernel='rbf')
x_kpca = kpca.fit_transform(x)
# 可视化
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
for i in range(1, 8):
    ax[0].scatter(
        x_kpca[y == i, 0], x_kpca[y == i, 1],
        color=colors[i-1], marker=markers[i-1], alpha=0.5
    )
    ax[1].scatter(
        x_kpca[y == i, 0], np.zeros((len(x_kpca[y == i, 0]), 1)) + 0.02,
        color=colors[i-1], marker=markers[i-1], alpha=0.5
    )
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()