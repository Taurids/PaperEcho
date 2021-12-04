import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据预处理
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv')  # 读取数据集
df['month'] = LabelEncoder().fit_transform(df['month'])  # 字符特征硬编码
df['day'] = LabelEncoder().fit_transform(df['day'])
x, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
sc = StandardScaler()  # 特征标准化
x = sc.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)  # 切分数据集

# KRR
krr = KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1)
krr.fit(X_train, y_train)
y_krr = krr.predict(X_test)
# 可视化
plt.plot(y_krr, color='red')
plt.plot(y_test, color='blue')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
