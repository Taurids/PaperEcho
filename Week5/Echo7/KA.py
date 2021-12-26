# coding=utf-8
import numpy as np


class Kernel:
    def __init__(self, kernel=None, sigma=None, theta=None):
        self.sigma = 1 if sigma is None else sigma
        self.theta = 1 if theta is None else theta
        if kernel == 'rbf':
            self.kernel = self.rbf
        elif kernel == 'sigmoid':
            self.kernel = self.sigmoid
        else:
            self.kernel = self.linear

    def linear(self, X1, X2):
        return np.matmul(X1, X2.T)

    def sigmoid(self, X1, X2):
        return np.tanh(np.matmul(X1, X2.T) + self.theta)

    def rbf(self, X1, X2, sigma=None):
        sigma = self.sigma if sigma is None else sigma
        mat = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return np.exp(-0.5 / sigma**2 * mat)

    def centering(self, K):
        m = K.shape[0]
        H = np.identity(m) - np.ones([m, m]) / m
        return np.dot(np.dot(H, K), H)

    # Centered Kernel Alignment
    def CKA(self, X, Y, center=True, K1_exist=False, K2_exist=False):
        K1 = self.kernel(X, X) if K1_exist is False else X
        K2 = self.kernel(Y, Y) if K2_exist is False else Y
        # Centering
        K1 = self.centering(K1) if center is True else K1
        K2 = self.centering(K2) if center is True else K2
        # Alignment Calculation
        HSIC = np.sum(K1 * K2)
        var1 = np.sqrt(np.sum(K1 * K1))
        var2 = np.sqrt(np.sum(K2 * K2))
        return HSIC / (var1 * var2)


if __name__ == '__main__':
    RX = np.random.randn(100, 64)
    RY = 2 * np.random.randint(2, size=(100, 1)) - 1

    K = Kernel()
    print('Linear CKA, between X and Y: {}'.format(K.CKA(RX, RY)))
    K = Kernel('rbf')
    print('RBF Kernel CKA, between X and Y: {}'.format(K.CKA(RX, RY)))
