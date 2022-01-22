from Echo7.kta import rbf
import numpy as np


def MMD(X, Y):
    n, m = X.shape[0], Y.shape[0]
    # Compute the Gram kernel matrix
    total = np.concatenate((X, Y), axis=0)
    kernels = rbf(total, total, gamma=0.5)
    # [ K_ss, K_st; K_ts, K_tt]
    XX = kernels[:n, :n]  # K_ss矩阵
    XY = kernels[:n, n:]  # K_st矩阵
    YX = kernels[n:, :n]  # K_ts矩阵
    YY = kernels[n:, n:]  # K_tt矩阵
    XX = np.sum(XX / (n * n), axis=1)  # K_ss/(ns*ns)
    XY = np.sum(XY / (n * m), axis=1)  # K_st/(ns*nt)
    YX = np.sum(YX / (m * n), axis=1)  # K_ts/(nt*ns)
    YY = np.sum(YY / (m * m), axis=1)  # K_tt/(nt*nt)

    loss = np.sum(XX - XY) + np.sum(YY - YX)
    return loss


if __name__ == "__main__":
    RX = np.random.normal(loc=0, scale=0.1, size=(100, 50))
    RY = np.random.normal(loc=0.1, scale=0.1, size=(100, 50))
    print("MMD Loss: ", MMD(RX, RY))
