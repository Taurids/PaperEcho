# coding=utf-8
import numpy as np
from numpy.matlib import repmat


# kernel function
def linear(X, Y):
    return np.matmul(X, Y.T)


def rbf(X, Y, gamma=0.5):
    mat = np.sum(X ** 2, 1).reshape(-1, 1) + np.sum(Y ** 2, 1) - 2 * np.dot(X, Y.T)
    return np.exp(-0.5 / gamma ** 2 * mat)


def sigmoid(X, Y, theta=1.0):
    return np.tanh(np.matmul(X, Y.T) + theta)


def ploy(X, Y, theta=1.0, p=3.0):
    return np.power(np.matmul(X, Y.T) + theta, p)


# kernel matrix
def pairwise_kernels(X, Y=None, kernel="linear", **kwargs):
    """Compute the kernel between arrays X and optional array Y.
    :return: Kernel(X, Y)
    """
    Y = X if Y is None else Y
    if kernel == 'rbf':
        gamma = kwargs['gamma'] if 'gamma' in kwargs.keys() else 0.5
        return rbf(X, Y, gamma=gamma)
    elif kernel == 'sigmoid':
        theta = kwargs['theta'] if 'theta' in kwargs.keys() else 1.0
        return sigmoid(X, Y, theta=theta)
    elif kernel == 'ploy':
        theta = kwargs['theta'] if 'theta' in kwargs.keys() else 1.0
        p = kwargs['p'] if 'p' in kwargs.keys() else 3.0
        return ploy(X, Y, theta=theta, p=p)
    else:
        return linear(X, Y)


# Kernel Alignment
def KA(KX, KY):
    # ---------- Inner product between matrices ----------------
    HS_IC = np.sum(KX * KY)  # Hilbert-Schmidt independence criterion
    var1 = np.sqrt(np.sum(KX * KX))
    var2 = np.sqrt(np.sum(KY * KY))
    # A = < KX, KY >F / ( || KX ||F, || KY ||F )
    return HS_IC / (var1 * var2)


# Kernel-Target Alignment
def KTA(X, y, kernel='linear', **kwargs):
    # --------------- get Original Alignment --------------------
    KX = pairwise_kernels(X, X, kernel=kernel, **kwargs)
    Ky = np.outer(y, y)
    # get Alignment
    ori_A = KA(KX, Ky)
    # --------------- get Optimal Alignment ---------------------
    # get vi
    _, eigenvector = np.linalg.eigh(KX)  # [4, 4]
    # get alpha
    LamLagrange = 0.6
    vyF2 = np.power(np.sum(eigenvector * y.T, axis=1), 2)  # <vi, y>_F^2  [4, 1]
    alpha = vyF2 / (2 * LamLagrange)  # alpha = <vi, y>_F^2 / (2 * lambda)
    alpha = alpha / np.sum(alpha)  # 0-1 normalization  [4, 1]
    # get Alignment
    W_alpha = np.sum(alpha * vyF2)
    opt_A = W_alpha / np.sqrt(np.sum(Ky * Ky))
    return ori_A, opt_A


# K -> HKH
def Centering(K):
    m, n = K.shape[0], K.shape[1]
    H0 = np.identity(m) - np.ones([m, m]) / m
    H1 = np.identity(n) - np.ones([n, n]) / n
    return np.dot(np.dot(H0, K), H1)


# Centered Kernel Alignment
def CKA(KX, KY, center=True):
    # Centering
    KX = Centering(KX) if center is True else KX
    KY = Centering(KY) if center is True else KY
    return KA(KX, KY)


def KernelAlignment(X, Y, kernel='linear', center=True, **params):
    KX, KY = pairwise_kernels(X, X, kernel, **params), pairwise_kernels(Y, Y, kernel, **params)
    return CKA(KX, KY, center)


def _get_mu_align(K_k, K_Y, gammas):
    # get mu = <K_k, K_Y>_F
    mu = []
    for i, gamma in enumerate(gammas):
        mu.append(CKA(K_k[i], K_Y))
    return mu


def _get_mu_alignf(K_k, K_Y, gammas):
    # get a and M
    a, M = [], []
    for i, _ in enumerate(gammas):
        a.append(np.sum(Centering(K_k[i]) * K_Y))
        M_kc = Centering(K_k[i])
        M_k = []
        for j, _ in enumerate(gammas):
            M_lc = Centering(K_k[j])
            M_kl = np.sum(M_kc * M_lc)
            M_k.append(M_kl)
        M.append(M_k)

    # mu = M^-1 a / || M^-1 a||
    tmp = np.dot(np.linalg.pinv(M), np.array([a]).T)
    mu = tmp / np.linalg.norm(tmp, ord=2)
    return mu


def combine_rbf_kernel(X, Y, K_Y, method, mu, gammas):
    # get K_k
    K_k = []
    for gamma in gammas:
        K_k.append(Centering(rbf(X, Y, gamma)).tolist())
    K_k = np.array(K_k)  # [2, 640, 640]

    # get mu
    if mu is None:
        if method == 'alignf':
            mu = _get_mu_alignf(K_k, K_Y, gammas)
        elif method == 'align':
            mu = _get_mu_align(K_k, K_Y, gammas)
        else:
            mu = [1 / len(gammas)] * len(gammas)  # 'unif'

    # K_mu = mu * K_k
    K_mu = np.zeros((K_k.shape[1], K_k.shape[2]))
    for i in range(len(gammas)):
        K_mu += mu[i] * K_k[i]
    return K_mu, mu


def pairwise_kernels_with_target(
        X, Y=None, K_Y=None, kernel="rbf", method=None, mu=None, gammas=None
):
    if not isinstance(gammas, list):
        gammas = list(gammas)
    return combine_rbf_kernel(X, Y, K_Y, method, mu, gammas)


if __name__ == '__main__':
    RX = np.random.randn(100, 64)
    RY = 2 * np.random.randint(2, size=(100, 1)) - 1

    RK = KernelAlignment(RX, RY)
    print('Linear CKA, between X and Y: {}'.format(RK))
    RK = KernelAlignment(RX, RY, 'rbf', gamma=0.9)
    print('RBF Kernel CKA, between X and Y: {}'.format(RK))
