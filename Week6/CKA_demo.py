# coding=utf-8
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from Echo7 import get_data, krr

# Parameters selection
data_uci = 'Splice'  # 'German', 'Ionosphere', 'Spambase', 'Splice'
method = 'alignf'  # 'unif', 'align', 'alignf'
gamma = [-9, -3]

# ----------------------------- Data Preprocess -----------------------------
X, y = get_data.fetch_uci('Splice', need_pre=True)
X_train, X_test, y_train, y_test = train_test_split(X, np.array([y]).T, train_size=800, test_size=200, random_state=7)
KF = KFold(n_splits=5)

err_rates, cka_scores = [], []
for KF_index, (train_index, valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index + 1, '折交叉验证开始...')
    # train valid split
    X_train_, X_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]

    clf = krr.KernelRidge(kernel='rbf', method='alignf', gamma=[-9, -3], REG=False)
    clf.fit(X_train_, y_train_)
    y_krr = clf.predict(X_valid_)

    cka_score = clf.get_CKA_score()
    err_rate = np.linalg.norm(y_krr - y_valid_, ord=1) / (2 * len(y_valid_))
    print('rbf Kernel CKA, between X and Y: {}'.format(cka_score))
    print('err rate in the validation is: {}'.format(err_rate))
    err_rates.append(err_rate)
    cka_scores.append(cka_score)
print(method + '实验最后平均输出结果：')
print('rbf Kernel CKA, between X and Y: {}'.format(np.mean(np.array(cka_scores))))
print('err rate in the validation is: {}'.format(np.mean(np.array(err_rates))))
