# coding=utf-8
import numpy as np
from .kta import pairwise_kernels, pairwise_kernels_with_target, CKA


class KernelRidge:
    def __init__(self, LamReg=.9, kernel="linear", method=None, gamma=None, REG=True):
        self.LamReg = LamReg  # lambda
        self.kernel = kernel
        self.gamma = gamma  # rbf kernel argument
        self.REG = REG  # False denote classification; True denote regression
        self.alpha = None
        self.X_train, self.K_Y = None, None
        self.K, self.mu = None, None
        self.method = method

    def _get_kernel(self, X, Y):
        if self.method is None:
            if isinstance(self.gamma, list) is False:
                return pairwise_kernels(X, Y, kernel=self.kernel, gamma=self.gamma)
            else:
                raise ValueError("gamma must be a single parameter, such as 0.5")
        else:
            return pairwise_kernels_with_target(
                X, Y, self.K_Y, kernel=self.kernel, method=self.method, mu=self.mu, gammas=self.gamma
            )

    def fit(self, X, y):
        self.X_train, self.K_Y = X, np.outer(y, y)

        if self.method is None:
            self.K = self._get_kernel(X, X)
        else:
            self.K, self.mu = self._get_kernel(X, X)
        tmp = np.linalg.pinv(self.K + self.LamReg)
        self.alpha = np.dot(tmp, y)  # [640, 1]

    def predict(self, X):
        if self.method is None:
            K = self._get_kernel(X, self.X_train)
        else:
            K, _ = self._get_kernel(X, self.X_train)
        y_krr = np.dot(K, self.alpha)  # Regression

        if not self.REG:  # Classification
            y_krr = np.tanh(y_krr)
            y_krr[y_krr >= 0] = 1
            y_krr[y_krr < 0] = -1
        return y_krr

    def get_CKA_score(self):
        return CKA(self.K, self.K_Y)

