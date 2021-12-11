import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 参数可调
_lambda = 0.7
_gamma = 0.3


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
def kernelRBF(X1, X2, gamma=1):
    RBF_all = []
    for i in range(X1.shape[0]):  # [104, 12]
        RBF_ = []
        for j in range(X2.shape[0]):  # [413, 12]
            _tmp = np.power(np.linalg.norm(X1[i] - X2[j]), 2)
            Kij = np.exp(-gamma * _tmp)
            RBF_.append(Kij)
        RBF_all.append(RBF_)
    return np.array(RBF_all)


K = kernelRBF(X_train, X_train)
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
