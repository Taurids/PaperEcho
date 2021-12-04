# coding=utf-8
import numpy as np
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt

# 数据预处理
(X_train, y_train), (X_test, y_test) = mnist.load_data()
x = np.append(X_train, X_test)
x = np.reshape(x, (70000, 28*28))
sc = StandardScaler()
x = sc.fit_transform(x)

# 调用 PCA 方法
kpca = KernelPCA(n_components=0.75, kernel='rbf', fit_inverse_transform=True)
X_reduced = kpca.fit_transform(x)
X_recovered = kpca.inverse_transform(X_reduced)

n = 5  # 显示的记录数
plt.figure(figsize=(10, 4))
for i in range(n):
    # 显示原始图片
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_train[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(10, 4))
for i in range(n):
    # 显示原始图片
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_recovered[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()