# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from Echo7.KA import Kernel
KA = Kernel('rbf')

# ------------------------------ Data Preprocess  -----------------------------------
# German
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
df = pd.read_csv(path, header=None)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
sc = StandardScaler()  # 特征标准化
X = sc.fit_transform(X)
y = np.array([2*y-1]).T
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.4)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, random_state=7, test_size=0.5)


# l2-KRR unif
def KRR_l2_unif_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_mu = 0
    for _sigma in p_base_kernel:
        K_mu += KA.rbf(X1, X1, sigma=_sigma)
    K_mu = lambda1 * K_mu / len(p_base_kernel)
    print('rbf Kernel CKA, between X and Y: {}'.format(KA.CKA(K_mu, y1, K1_exist=True)))
    tmp_KRR = np.linalg.pinv(K_mu + lambda2)  # [600, 600]
    alpha_KRR = np.dot(tmp_KRR, y1)  # [600, 1]

    K_mu = 0
    for _sigma in p_base_kernel:
        K_mu += KA.rbf(X2, X1, sigma=_sigma)
    K_mu = lambda1 * K_mu / len(p_base_kernel)
    y_krr = np.dot(K_mu, alpha_KRR)
    y_krr = np.tanh(y_krr)
    y_krr[y_krr >= 0] = 1
    y_krr[y_krr < 0] = -1
    return np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))


err_rate = KRR_l2_unif_predict(X_train, X_valid, y_train, y_valid, p_base_kernel=[-12, -7])
print('err rate in the validation is: {}'.format(err_rate))


# l2-KRR align
def KRR_l2_align_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_mu = 0
    for _sigma in p_base_kernel:
        K_k = KA.rbf(X1, X1, sigma=_sigma)
        mu_k = KA.CKA(K_k, y1, K1_exist=True)
        K_mu += mu_k * K_k
    K_mu = K_mu * lambda1
    print('rbf Kernel CKA, between X and Y: {}'.format(KA.CKA(K_mu, y1, K1_exist=True)))
    tmp_KRR = np.linalg.pinv(K_mu + lambda2)  # [600, 600]
    alpha_KRR = np.dot(tmp_KRR, y1)  # [600, 1]

    K_mu = 0
    for _sigma in p_base_kernel:
        K_k = KA.rbf(X1, X1, sigma=_sigma)  # [200, 600] [600, 1]
        mu_k = KA.CKA(K_k, y1, K1_exist=True)
        K_k_hat = KA.rbf(X2, X1, sigma=_sigma)
        K_mu += mu_k * K_k_hat
    K_mu = K_mu * lambda1
    y_krr = np.dot(K_mu, alpha_KRR)
    y_krr = np.tanh(y_krr)
    y_krr[y_krr >= 0] = 1
    y_krr[y_krr < 0] = -1
    return np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))
0

err_rate = KRR_l2_align_predict(X_train, X_valid, y_train, y_valid, p_base_kernel=[-12, -7])
print('err rate in the validation is: {}'.format(err_rate))


# l2-KRR alignf
def KRR_l2_alignf_predict(X1, X2, y1, y2, lambda2=0.3, p_base_kernel=None):
    a = []
    for _sigma in p_base_kernel:
        K_k = KA.rbf(X1, X1, sigma=_sigma)
        a.append(np.sum(KA.centering(K_k) * KA.linear(y1, y1)))
    M = []
    for _sigma_k in p_base_kernel:
        K_kc = KA.centering(KA.rbf(X1, X1, sigma=_sigma_k))
        M_k = []
        for _sigma_l in p_base_kernel:
            K_lc = KA.centering(KA.rbf(X1, X1, sigma=_sigma_l))
            M_kl = np.sum(K_kc*K_lc)
            M_k.append(M_kl)
        M.append(M_k)
    tmp = np.dot(np.linalg.pinv(M), np.array([a]).T)
    mu = tmp / np.linalg.norm(tmp, ord=2)
    K_mu = 0
    for k, _sigma in enumerate(p_base_kernel):
        K_mu += mu[k] * KA.rbf(X1, X1, sigma=_sigma)
    print('rbf Kernel CKA, between X and Y: {}'.format(KA.CKA(K_mu, y1, K1_exist=True)))
    tmp_KRR = np.linalg.pinv(K_mu + lambda2)  # [600, 600]
    alpha_KRR = np.dot(tmp_KRR, y1)  # [600, 1]

    K_mu = 0
    for k, _sigma in enumerate(p_base_kernel):
        K_mu += mu[k] * KA.rbf(X2, X1, sigma=_sigma)
    y_krr = np.dot(K_mu, alpha_KRR)
    y_krr = np.tanh(y_krr)
    y_krr[y_krr >= 0] = 1
    y_krr[y_krr < 0] = -1
    return np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))


err_rate = KRR_l2_alignf_predict(X_train, X_valid, y_train, y_valid, p_base_kernel=[-12, -7])
print('err rate in the validation is: {}'.format(err_rate))


def KRR_l2_1_stage_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_0 = KA.rbf(X1, X1)
    mu = [0] * len(p_base_kernel)
    alpha = np.dot(np.linalg.pinv(K_0 + lambda2), y1)
    for _ in range(10):
        v = []
        for _sigma in p_base_kernel:
            tmp = np.dot(np.dot(alpha.T, KA.rbf(X1, X1, sigma=_sigma)), alpha)
            v.extend(tmp)
        v = np.array(v).T
        mu += lambda1 * v[0] / np.linalg.norm(v[0], ord=2)
        K_mu = 0
        for k, _sigma in enumerate(p_base_kernel):
            K_mu += mu[k] * KA.rbf(X1, X1, sigma=_sigma)
        alpha = 0.5 * alpha + 0.5 * np.dot(np.linalg.pinv(K_mu + lambda2), y1)
        print('rbf Kernel CKA, between X and Y: {}'.format(KA.CKA(K_mu, y1, K1_exist=True)))

    # 预测
    K_mu = 0
    for k, _sigma in enumerate(p_base_kernel):
        K_mu += mu[k] * KA.rbf(X2, X1, sigma=_sigma)
    y_krr = np.dot(K_mu, alpha)
    y_krr = np.tanh(y_krr)
    y_krr[y_krr >= 0] = 1
    y_krr[y_krr < 0] = -1
    return np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))


err_rate = KRR_l2_1_stage_predict(X_train, X_valid, y_train, y_valid, p_base_kernel=[-12, -7])
print('err rate in the validation is: {}'.format(err_rate))
