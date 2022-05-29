import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 参数可调
_lambda = 0.7
_sigma = 0.3


# 数据预处理
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')  # 读取数据集
df['month'] = LabelEncoder().fit_transform(df['month'])  # 字符特征硬编码
df['day'] = LabelEncoder().fit_transform(df['day'])
x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
sc = StandardScaler()  # 特征标准化
x = sc.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)  # 切分数据集

# RR
tmp = np.linalg.pinv(np.dot(X_train.T, X_train) + _lambda)
alpha = np.dot(np.dot(tmp, X_train.T), y_train)
y_rr = np.dot(X_test, alpha)


# KRR
def kernelRBF(X1, X2, sigma=1.0):
    mat = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
    return np.exp(-0.5 / sigma ** 2 * mat)


K = kernelRBF(X_train, X_train, _sigma)
# Centralize the kernel matrix
N = K.shape[0]
one_n = np.ones((N, N)) / N
K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

tmp = np.linalg.pinv(K + _lambda)  # [413, 413]
alpha = np.dot(tmp, y_train)  # [413,]

# 预测
K = kernelRBF(X_test, X_train)
y_krr = np.dot(K, alpha.T)

# 可视化
plt.plot(y_rr, color='lightgreen')
plt.plot(y_krr, color='red')
plt.plot(y_test, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
