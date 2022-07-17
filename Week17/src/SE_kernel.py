import torch


def SE_kernel_multi(sample1, sample2, **kwargs):
    """
    Compute the multidim square exponential kernel
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper-parameter:bandwidth
    :return:
    """
    bandwidth = kwargs['kernel_hyper']['bandwidth']
    if len(sample1.shape) == 4:  # * x N x d
        bandwidth = bandwidth.unsqueeze(-1).unsqueeze(-1)

    sample_diff = sample1 - sample2  # N x N x dim

    norm_sample = torch.norm(sample_diff, dim=-1) ** 2  # N x N or * x N x N

    K = torch.exp(-norm_sample / (bandwidth ** 2 + 1e-9))
    return K


def trace_SE_kernel_multi(sample1, sample2, **kwargs):
    # compute trace of second-order derivative of RBF kernel
    """
    Compute the trace of 2 order gradient of K
    :param sample1: x : N x N x dim
    :param sample2: y : N x N x dim
    :param kwargs: kernel hyper: bandwidth
    :return:
    """
    bandwidth = kwargs['kernel_hyper']['bandwidth']
    K = kwargs['K']

    diff = sample1 - sample2  # N x N x dim
    H = K * (2. / (bandwidth ** 2 + 1e-9) * sample1.shape[-1] - 4. / (bandwidth ** 4 + 1e-9)
             * torch.sum(diff * diff, dim=-1))  # N x N
    return H
