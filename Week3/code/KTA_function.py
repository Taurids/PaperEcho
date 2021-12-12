# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def KTAlignment(yk, mu=1):
    # Original
    K = mu * np.dot(np.array([yk]).T, np.array([yk]))
    yy = np.dot(np.array([yk]).T, np.array([yk]))
    KF = np.linalg.norm(np.power(K, 2), ord='fro')
    yyF = np.linalg.norm(np.power(yy, 2), ord='fro')  # m
    KyyF = np.sum(np.multiply(K, yy), axis=(0, 1))
    _ori_A = KyyF / (KF * yyF)
    # Optimal
    eigenvalue, eigenvector = np.linalg.eigh(K)
    vy4 = 0
    for i in range(len(eigenvector)):
        vy = np.sum(np.multiply(np.array([eigenvector[:, i]]), np.array([yk])), axis=(0, 1))
        vy4 += np.power(vy, 4)
    _opt_A = np.sqrt(vy4) / yyF
    return _ori_A, _opt_A


if __name__ == '__main__':
    # ------------------------------ Data Preprocess  -----------------------------------
    path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
    df = pd.read_csv(path, header=None)
    X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values
    # StandardScaler
    x_means = np.array(np.mean(np.mat(X), axis=0))
    x_stds = np.array(np.std(np.mat(X), axis=0))
    X = (X - x_means) / x_stds
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7, test_size=0.2)

    # ------------------------------ Kernel-Target Alignment -----------------------------
    ori_A, opt_A = KTAlignment(y_train, mu=2)
    print('Original Train Alignment: ', ori_A)
    print('Optimal Train Alignment: ', opt_A)
    ori_A, opt_A = KTAlignment(y_test, mu=2)
    print('Original Test Alignment: ', ori_A)
    print('Optimal Test Alignment: ', opt_A)

