import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def kernelRBF(X1, X2, gamma=1):
    RBF_all = []
    for i in range(X1.shape[0]):
        RBF_ = []
        for j in range(X2.shape[0]):
            _tmp = np.power(np.linalg.norm(X1[i] - X2[j]), 2)
            Kij = np.exp(-gamma * _tmp)
            RBF_.append(Kij)
        RBF_all.append(RBF_)
    return np.array(RBF_all)


# 参数可调
_lambda = 0.7
_gamma = 0.3

# ------------- 数据预处理 ---------------------
# 读取数据集
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
x_list = []
for i in range(29):
    if i not in [0, 21, 23]:
        x_list.append(i)
X, y_T, _ = df.iloc[:, x_list].values, df.iloc[:, 21].values, df.iloc[:, 23].values
# 特征标准化
sc = StandardScaler()
X = sc.fit_transform(X)
# 切分数据集
X_train_T, X_test_T, y_train_T, y_test_T = train_test_split(X, y_T, test_size=0.2)


# ------------------------------ KRR ----------------------------------------
# KRR 温度T
K = kernelRBF(X_train_T, X_train_T)
tmp = np.linalg.pinv(K + _lambda)  # [500, 500]
alpha = np.dot(tmp, y_train_T)  # [500,]

# 预测
K = kernelRBF(X_test_T, X_train_T)
yT_krr = np.dot(K, alpha.T)

# 可视化
plt.plot(yT_krr, color='red')
plt.plot(y_test_T, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
