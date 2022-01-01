# coding=utf-8
import numpy as np
from .kta import pairwise_kernels, pairwise_kernels_with_target, Centering, CKA


class KernelRidge:
    def __init__(self, alpha=.9, kernel="linear", method=None,
                 gamma=None, theta=None, classify=True, kernel_params=None):

        self.alpha = alpha  # lambda
        self.kernel = kernel
        self.gamma = gamma  # rbf kernel arg
        self.theta = theta  # sigmoid kernel arg
        self.classify = classify  # 0 denote classification; 1 denote regression
        self.alpha_ = None
        self.X_train = None
        self.y_train = None
        self.K_mu, self.mu = None, None
        self.kernel_params = kernel_params  # all kernel arg
        self.method = method

    def _get_kernel(self, X, Y, target=None, mu=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "theta": self.theta}
        if target is None:
            return Centering(pairwise_kernels(X, Y, kernel=self.kernel, **params))
        return pairwise_kernels_with_target(
            X, Y, target, kernel=self.kernel, method=self.method, mu=mu, **params
        )

    def fit(self, X, y):
        if self.method == 'unif':
            self.K_mu = self._get_kernel(X, X)
        else:
            self.K_mu, self.mu = self._get_kernel(X, X, target=y)
        tmp = np.linalg.pinv(self.K_mu + self.alpha)
        self.alpha_ = np.dot(tmp, y)  # [640, 1]

        self.X_train, self.y_train = X, y

    def predict(self, X):
        if self.method == 'unif':
            K = self._get_kernel(X, self.X_train)
        else:
            K = self._get_kernel(X, self.X_train, mu=self.mu)
        if self.classify:  # Classification
            y_krr = np.tanh(np.dot(K, self.alpha_))
            y_krr[y_krr >= 0] = 1
            y_krr[y_krr < 0] = -1
        else:  # Regression
            y_krr = np.dot(K, self.alpha_)
        return y_krr

    def get_CKA_score(self):
        K_Y = np.outer(self.y_train, self.y_train)
        return CKA(self.K_mu, K_Y)

