# coding=utf-8
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt


def plot_digits(X, title):
    """Small helper function to plot 25 digits."""
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(5, 5))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((28, 28)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=15)


# ------------- Data Preprocess -------------------------
# Load MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
# normalize the dataset such that all pixel values are in the range (0, 1).
X = MinMaxScaler().fit_transform(X)
# split the dataset into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=7, train_size=2000, test_size=100)
# add a Gaussian noise to create a new dataset
rng = np.random.RandomState(7)
noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise

# --------- Learn the PCA basis --------------------
pca = PCA(n_components=154)
kernel_pca = KernelPCA(n_components=154, kernel="rbf", alpha=5e-3, gamma=1e-3, fit_inverse_transform=True)
pca.fit(X_train_noisy)
kernel_pca.fit(X_train_noisy)

# --------- Reconstruct and denoise test images --------------------
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(kernel_pca.transform(X_test_noisy))

# ---------- Data Visualization ------------------------------------
# see the difference among noise-free images, noisy images and denoised images
plot_digits(X_test, "Uncorrupted test images")
plot_digits(X_test_noisy, f"Noisy test images\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}")
plot_digits(
    X_reconstructed_pca,
    f"PCA reconstruction\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
)
plot_digits(
    X_reconstructed_kernel_pca,
    "Kernel PCA reconstruction\n"
    f"MSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}",
)
plt.show()
