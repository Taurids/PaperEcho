# coding=utf-8
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def kernels_RBF(X1, X2, _gamma=1):
    RBF_all = []
    for i in range(X1.shape[0]):
        RBF_ = []
        for j in range(X2.shape[0]):
            _tmp = np.power(np.linalg.norm(X1[i] - X2[j]), 2)
            Kij = np.exp(-_gamma * _tmp)
            RBF_.append(Kij)
        RBF_all.append(RBF_)
    return np.array(RBF_all)


def kernels_Sigmoid(X1, X2, _theta=1):
    return np.tanh(np.matmul(X1, X2.T) + _theta)


def kernels_POLY(X1, X2, _theta=1, _p=3):
    return np.power(np.matmul(X1, X2.T) + _theta, _p)


def kernels_LINEAR(X1, X2):
    return np.matmul(X1, X2.T)


def kernels_target_alignment(Mat_X, Mat_Y):
    # ------------- Original ---------------------
    # Kernel matrices
    KX = kernels_LINEAR(Mat_X, Mat_X)
    KY = np.outer(np.array(Mat_Y), np.array(Mat_Y))
    # < KX, KY >F
    KX_KY_F = np.sum(np.multiply(KX, KY), axis=(0, 1))
    # || KX ||F
    KX_F = np.linalg.norm(np.power(KX, 2), ord='fro')
    # || KY ||F
    KY_F = np.linalg.norm(np.power(KY, 2), ord='fro')
    # A = < KX, KY >F / ( || KX ||F, || KY ||F )
    Ori_Align = KX_KY_F / (KX_F * KY_F)

    # ------------ Optimal -----------------------
    # give K = sum_i (alpha_i vi vi_T)
    # the base kernels Ki = vi vi_T
    eigenvalue, eigenvector = np.linalg.eigh(KX)  # all vi
    # finding the optimal alpha
    v_y_sum = 0
    for i in range(len(eigenvector)):
        v_y_F2 = np.sum(np.multiply(np.array([eigenvector[:, i]]), np.array([Mat_Y])), axis=(0, 1))
        v_y_sum += np.power(v_y_F2, 4)
    # A = sqrt( v_y_sum ) / || KY ||F
    Opt_Align = np.sqrt(v_y_sum) / KY_F

    return Ori_Align, Opt_Align


if __name__ == '__main__':
    X = np.random.random((4, 2))
    Y = np.array([-1, -1, 1, 1])
    ori_A, opt_A = kernels_target_alignment(X, Y)
    print('Original Alignment: ', ori_A)
    print('Optimal Alignment: ', opt_A)

