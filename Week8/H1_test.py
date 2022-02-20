import numpy as np
from Echo7 import kta
from tqdm import tqdm
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
sns.set()


m, n = 1000, 800
t = m + n
c = 88
rhoX, rhoY = m / t, n / t

MMDu2_list = []
RX = np.random.laplace(loc=0, scale=1, size=(m, 7))
RY = np.random.laplace(loc=t**(-1/2)*c, scale=1, size=(n, 7))
for i in tqdm(range(2000)):
    Kxx = kta.Centering(kta.rbf(RX, RX, gamma=0.5))
    Kyy = kta.Centering(kta.rbf(RY, RY, gamma=0.5))
    Kxy = kta.Centering(kta.rbf(RX, RY, gamma=0.5))
    eigenvalue_Kxx, _ = np.linalg.eigh(Kxx)
    eigenvalue_Kyy, _ = np.linalg.eigh(Kyy)
    U, eigenvalue_Kxy, VT = np.linalg.svd(Kxy)

    step1 = step2 = step3 = 0
    al = np.random.normal(loc=0, scale=1, size=len(eigenvalue_Kxx))
    bl = np.random.normal(loc=0, scale=1, size=len(eigenvalue_Kyy))
    for j, value in enumerate(eigenvalue_Kxx):
        step1 = 1 / rhoX * value * (np.power(al[j], 2) - 1)
    for j, value in enumerate(eigenvalue_Kyy):
        step2 = 1 / rhoY * value * (np.power(bl[j], 2) - 1)
    for j, value in enumerate(eigenvalue_Kxy):
        step3 = 2 / np.sqrt(rhoX * rhoY) * value * al[j] * bl[j]
    tMMDc2 = step1 + step2 - step3
    step4 = 2 * c * np.power(rhoX, -0.5) * al[0]
    step5 = 2 * c * np.power(rhoY, -0.5) * bl[0]

    tMMDu2 = tMMDc2 + step4 - step5 + c**2
    MMDu2_list.append(tMMDu2 / t)


plt.figure(figsize=(8, 4))
sns.distplot(MMDu2_list, bins=100)
# 用正态分布拟合
plt.legend()
plt.grid(linestyle='--')
plt.show()

