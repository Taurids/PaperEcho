# coding=utf-8
import argparse
import numpy as np
from scipy.spatial import distance
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

my_parser = argparse.ArgumentParser()
# Required parameters
my_parser.add_argument("--n_components", default=154, type=int, help="")

args = my_parser.parse_args()


def plot_digits(X, title):
    """Small helper function to plot 25 digits."""
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(5, 5))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((28, 28)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=15)


def kernelRBF(X1, X2, _gamma=1):
    RBF_all = []
    for i in range(X1.shape[0]):  # [104, 12]
        RBF_ = []
        for j in range(X2.shape[0]):  # [413, 12]
            _tmp = np.power(np.linalg.norm(X1[i] - X2[j]), 2)
            Kij = np.exp(-_gamma * _tmp)
            RBF_.append(Kij)
        RBF_all.append(RBF_)
    return np.array(RBF_all)


# ------------- Data Preprocess -------------------------
# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# normalize the dataset such that all pixel values are in the range (0, 1).
X = MinMaxScaler().fit_transform(X)
# split the dataset into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7, train_size=1000, test_size=100)
# add a Gaussian noise to create a new dataset
rng = np.random.RandomState(7)
noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise

# -------------------------------- PCA method --------------------------------------
C = np.matmul(X_train_noisy.T, X_train_noisy) / X_train_noisy.shape[0]
_, eigenvector = np.linalg.eigh(C)
eigenvector = eigenvector[:, ::-1][:, :args.n_components]
x_pca = np.matmul(X_test_noisy, eigenvector)
X_reconstructed_pca = np.matmul(x_pca, eigenvector.T)

# -------------------------------- KPCA method --------------------------------------
# Compute the kernel matrix
gamma = 1
sq_dists = distance.pdist(X_train, 'sqeuclidean')
mat_sq_dists = distance.squareform(sq_dists)
K = np.exp(-gamma * mat_sq_dists)

# Centralize the kernel matrix
N = K.shape[0]
one_n = np.ones((N, N)) / N
K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

eigenvalue, eigenvector = np.linalg.eigh(K)
eigenvalue = np.sqrt(eigenvalue[::-1])  # [500, 154]

eigenvector = eigenvector[:, ::-1][:, :args.n_components]  # [500, 154]
vi = np.divide(eigenvector, eigenvalue.reshape(-1, 1))
K = kernelRBF(X_test, X_train)
x_kpca = np.dot(K, vi)
# how to reconstruct the kernel PCA ?

# ----------------------------- Data Visualization ---------------------------------
# see the difference among noise-free images, noisy images and denoised images
plot_digits(X_test, "Uncorrupted test images")
plot_digits(X_test_noisy, f"Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}")
plot_digits(
    X_reconstructed_pca,
    f"PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
)
# plot_digits(
#     X_reconstructed_kpca,
#     "Kernel PCA reconstruction\n"
#     f"MSE: {np.mean((X_test - X_reconstructed_kpca) ** 2):.2f}",
# )
plt.show()
