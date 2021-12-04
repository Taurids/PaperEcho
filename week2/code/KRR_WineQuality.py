import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# ------------- 数据预处理 ---------------------
# 读取数据集
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv')
x_list = []
for i in range(29):
    if i not in [0, 21, 23]:
        x_list.append(i)
x, y_T, y_RH = df.iloc[:, x_list].values, df.iloc[:, 21].values, df.iloc[:, 23].values
# 特征标准化
sc = StandardScaler()
x = sc.fit_transform(x)
# 切分数据集
test_size = 4000
X_train, X_test = x[test_size:], x[:test_size]
yT_train, yT_test = y_T[test_size:], y_T[:test_size]
yRH_train, yRH_test = y_RH[test_size:], y_RH[:test_size]

# KRR 温度T
krr = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid={"alpha": [1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
)
krr.fit(X_train, yT_train)
yT_krr = krr.predict(X_test)

# KRR 湿度RH
krr = GridSearchCV(
    KernelRidge(kernel="rbf"),
    param_grid={"alpha": [1e0, 1e-1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)}
)
krr.fit(X_train, yRH_train)
yRH_krr = krr.predict(X_test)

# 可视化
_, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
ax[0].plot(yT_krr, color='red')
ax[0].plot(yT_test, color='blue')
ax[0].set_xlabel('X_T')
ax[0].set_ylabel('Y_T')
ax[1].plot(yRH_krr, color='red')
ax[1].plot(yRH_test, color='blue')
ax[1].set_xlabel('X_RH')
ax[1].set_ylabel('Y_RH')
plt.show()
