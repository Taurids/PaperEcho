import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# 数据预处理
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
x, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
sc = StandardScaler()
x = sc.fit_transform(x)

# PCA
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x)
# 可视化
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax[0].scatter(x_pca[y == 1, 0], x_pca[y == 1, 1], color='red', marker='^', alpha=0.5)
ax[0].scatter(x_pca[y == 2, 0], x_pca[y == 2, 1], color='blue', marker='o', alpha=0.5)
ax[0].scatter(x_pca[y == 3, 0], x_pca[y == 3, 1], color='lightgreen', marker='s', alpha=0.5)
ax[1].scatter(x_pca[y == 1, 0], np.zeros((len(x_pca[y == 1, 0]), 1))+0.02, color='red', marker='^', alpha=0.5)
ax[1].scatter(x_pca[y == 2, 0], np.zeros((len(x_pca[y == 2, 0]), 1))+0.02, color='blue', marker='o', alpha=0.5)
ax[1].scatter(x_pca[y == 3, 0], np.zeros((len(x_pca[y == 3, 0]), 1))+0.02, color='lightgreen', marker='s', alpha=0.5)
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