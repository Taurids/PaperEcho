# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from Echo7.KA import Kernel
KA = Kernel('rbf')

# ------------------------------ Data Preprocess  -----------------------------------
path = './kin-family/kin-8nm.data'
df = pd.read_csv(path, header=None)
df = df[0].str.split('  ', expand=True)
df.drop(0, axis=1, inplace=True)
for i in range(1, 10):
    df[i] = df[i].astype(float)
X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
sc = StandardScaler()  # 特征标准化
X = sc.fit_transform(X)
y = np.array([y]).T

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=800, test_size=200)
KF = KFold(n_splits=5)


# l2-KRR unif
def KRR_l2_unif_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_mu = 0
    for _sigma in p_base_kernel:
        K_mu += KA.rbf(X1, X1, sigma=_sigma)
    K_mu = lambda1 * K_mu / len(p_base_kernel)
    CKA_score = KA.CKA(K_mu, y1, K1_exist=True)
    print('rbf Kernel CKA, between X and Y: {}'.format(CKA_score))
    tmp_KRR = np.linalg.pinv(K_mu + lambda2)  # [600, 600]
    alpha_KRR = np.dot(tmp_KRR, y1)  # [600, 1]

    K_mu = 0
    for _sigma in p_base_kernel:
        K_mu += KA.rbf(X2, X1, sigma=_sigma)
    K_mu = lambda1 * K_mu / len(p_base_kernel)
    y_krr = np.dot(K_mu, alpha_KRR)
    krr_err_rate = np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))
    return krr_err_rate, CKA_score


err_rates, cka_scores = [], []
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # 训练集划分
    X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    err_rate, cka_score = KRR_l2_unif_predict(X_train_, X_valid_, y_train_, y_valid_, p_base_kernel=[-3, 3])
    print('err rate in the validation is: {}'.format(err_rate))
    err_rates.append(err_rate)
    cka_scores.append(cka_score)
print('unif实验最后平均输出结果：')
print('rbf Kernel CKA, between X and Y: {}'.format(np.mean(np.array(cka_scores))))
print('err rate in the validation is: {}'.format(np.mean(np.array(err_rates))))


# l2-KRR align
def KRR_l2_align_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_mu = 0
    for _sigma in p_base_kernel:
        K_k = KA.rbf(X1, X1, sigma=_sigma)
        mu_k = KA.CKA(K_k, y1, K1_exist=True)
        K_mu += mu_k * K_k
    K_mu = K_mu * lambda1
    CKA_score = KA.CKA(K_mu, y1, K1_exist=True)
    # print('rbf Kernel CKA, between X and Y: {}'.format(CKA_score))
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
    krr_err_rate = np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))
    return krr_err_rate, CKA_score


err_rates, cka_scores = [], []
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # 训练集划分
    X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    err_rate, cka_score = KRR_l2_align_predict(X_train_, X_valid_, y_train_, y_valid_, p_base_kernel=[-3, 3])
    # print('err rate in the validation is: {}'.format(err_rate))
    err_rates.append(err_rate)
    cka_scores.append(cka_score)
print('align实验最后平均输出结果：')
print('rbf Kernel CKA, between X and Y: {}'.format(np.mean(np.array(cka_scores))))
print('err rate in the validation is: {}'.format(np.mean(np.array(err_rates))))


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
    CKA_score = KA.CKA(K_mu, y1, K1_exist=True)
    # print('rbf Kernel CKA, between X and Y: {}'.format(CKA_score))
    tmp_KRR = np.linalg.pinv(K_mu + lambda2)  # [600, 600]
    alpha_KRR = np.dot(tmp_KRR, y1)  # [600, 1]

    K_mu = 0
    for k, _sigma in enumerate(p_base_kernel):
        K_mu += mu[k] * KA.rbf(X2, X1, sigma=_sigma)
    y_krr = np.dot(K_mu, alpha_KRR)
    krr_err_rate = np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))
    return krr_err_rate, CKA_score


err_rates, cka_scores = [], []
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # 训练集划分
    X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    err_rate, cka_score = KRR_l2_alignf_predict(X_train_, X_valid_, y_train_, y_valid_, p_base_kernel=[-3, 3])
    # print('err rate in the validation is: {}'.format(err_rate))
    err_rates.append(err_rate)
    cka_scores.append(cka_score)
print('alignf实验最后平均输出结果：')
print('rbf Kernel CKA, between X and Y: {}'.format(np.mean(np.array(cka_scores))))
print('err rate in the validation is: {}'.format(np.mean(np.array(err_rates))))


def KRR_l2_1_stage_predict(X1, X2, y1, y2, lambda1=2, lambda2=0.7, p_base_kernel=None):
    K_0 = KA.rbf(X1, X1)
    mu = [0] * len(p_base_kernel)
    alpha = np.dot(np.linalg.pinv(K_0 + lambda2), y1)
    CKA_score = 0
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
        CKA_score = KA.CKA(K_mu, y1, K1_exist=True)
        # print('rbf Kernel CKA, between X and Y: {}'.format(CKA_score))

    # 预测
    K_mu = 0
    for k, _sigma in enumerate(p_base_kernel):
        K_mu += mu[k] * KA.rbf(X2, X1, sigma=_sigma)
    y_krr = np.dot(K_mu, alpha)
    krr_err_rate = np.linalg.norm(y_krr - y2, ord=1) / (2 * len(y2))
    return krr_err_rate, CKA_score


err_rates, cka_scores = [], []
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # 训练集划分
    X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    err_rate, cka_score = KRR_l2_1_stage_predict(X_train_, X_valid_, y_train_, y_valid_, p_base_kernel=[-3, 3])
    # print('err rate in the validation is: {}'.format(err_rate))
    err_rates.append(err_rate)
    cka_scores.append(cka_score)
print('1_stage实验最后平均输出结果：')
print('rbf Kernel CKA, between X and Y: {}'.format(np.mean(np.array(cka_scores))))
print('err rate in the validation is: {}'.format(np.mean(np.array(err_rates))))
