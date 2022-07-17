import torch
import numpy as np


def compute_KSD(samples1, samples2, score_func, kernel, trace_kernel, **kwargs):
    # Compute KSD
    if 'flag_U' in kwargs:
        flag_U = kwargs['flag_U']
    else:
        flag_U = False

    if 'flag_retain' in kwargs:
        flag_retain = kwargs['flag_retain']
        flag_create = kwargs['flag_create']
    else:
        flag_retain = False
        flag_create = False

    kernel_hyper = kwargs['kernel_hyper']
    divergence_accum = 0

    samples1_crop_exp = torch.unsqueeze(samples1, dim=1).repeat(1, samples2.shape[0], 1)  # N × N(rep) × dim
    samples2_crop_exp = torch.unsqueeze(samples2, dim=0).repeat(samples1.shape[0], 1, 1)  # N(rep) × N × dim

    # Compute Term1
    if 'score_sample1' in kwargs:
        # compute score outside of this function
        score_sample1 = kwargs['score_sample1']
        score_sample2 = kwargs['score_sample2']
    else:
        score_sample1 = score_func(samples1)  # N
        score_sample1 = torch.autograd.grad(torch.sum(score_sample1), samples1)[0]  # N x dim
        score_sample2 = score_func(samples2)  # N
        score_sample2 = torch.autograd.grad(torch.sum(score_sample2), samples2)[0]  # N x dim

    score_sample1_exp = torch.unsqueeze(score_sample1, dim=1)  # N x 1 x dim
    score_sample2_exp = torch.unsqueeze(score_sample2, dim=0)  # 1 x N x dim

    K = kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper)

    if flag_U:
        Term1 = (K - torch.diag(torch.diag(K))) * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N
    else:
        Term1 = K * torch.sum(score_sample1_exp * score_sample2_exp, dim=-1)  # N x N

    # Compute Term 2, directly use autograd for kernel gradient
    if flag_U:
        grad_K_2 = torch.autograd.grad(
            torch.sum((K - torch.diag(torch.diag(K)))), samples2_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create
        )[0]  # N x N x dim
    else:
        grad_K_2 = torch.autograd.grad(
            torch.sum(K), samples2_crop_exp, retain_graph=flag_retain, create_graph=flag_create
        )[0]  # N x N x dim
    Term2 = torch.sum(score_sample1_exp * grad_K_2, dim=-1)  # N x N

    # Compute Term 3
    if flag_U:
        K = kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper)
        grad_K_1 = torch.autograd.grad(
            torch.sum((K - torch.diag(torch.diag(K)))), samples1_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create
        )[0]  # N x N x dim
    else:
        K = kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper)
        grad_K_1 = torch.autograd.grad(
            torch.sum(K), samples1_crop_exp,
            retain_graph=flag_retain, create_graph=flag_create
        )[0]  # N x N x dim
    Term3 = torch.sum(score_sample2_exp * grad_K_1, dim=-1)  # N x N

    # Compute Term 4, manually derive the trace of high-order derivative of kernel called trace_kernel
    K = kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper)
    if flag_U:
        T_K = trace_kernel(
            samples1_crop_exp, samples2_crop_exp,
            kernel_hyper=kernel_hyper, K=K - torch.diag(torch.diag(K))
        )
        grad_K_12 = T_K  # N x N
    else:
        T_K = trace_kernel(samples1_crop_exp, samples2_crop_exp, kernel_hyper=kernel_hyper, K=K)
        grad_K_12 = T_K  # N x N
    Term4 = grad_K_12

    KSD_comp = torch.sum(Term1 + 1 * Term2 + 1 * Term3 + 1 * Term4)

    divergence_accum += KSD_comp

    if flag_U:
        KSD = divergence_accum / ((samples1.shape[0] - 1) * samples2.shape[0])
    else:
        KSD = divergence_accum / (samples1.shape[0] * samples2.shape[0])

    return KSD, Term1 + Term2 + Term3 + Term4


def bootstrap_KSD(n_boots, p_x, q_samples, kernel, trace_kernel, kernel_hyper):
    n_sample = q_samples.shape[0]

    weights = np.random.multinomial(n_sample, np.ones(n_sample) / n_sample, size=int(n_boots))
    weights = weights / n_sample

    weights = torch.from_numpy(weights).type(q_samples.type())

    # Compute KSD
    KSD, KSD_comp = compute_KSD(
        q_samples, q_samples.clone().detach().requires_grad_(), p_x.log_prob,
        kernel, trace_kernel, flag_U=True, kernel_hyper=kernel_hyper
    )
    with torch.no_grad():
        # now compute boots strap samples
        KSD_comp = torch.unsqueeze(KSD_comp, dim=0)  # 1 x N x N

        weights_exp = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x n

        bootstrap_samples = (weights_exp - 1. / n_sample) * KSD_comp * (weights_exp2 - 1. / n_sample)  # m x N x N
        bootstrap_samples = torch.sum(torch.sum(bootstrap_samples, dim=-1), dim=-1)

    return KSD, bootstrap_samples


def compute_MMD(samples1, samples2, kernel, kernel_hyper, flag_U=True, flag_simple_U=True):
    # samples1: N x dim
    # samples2: N x dim
    n = samples1.shape[0]
    m = samples2.shape[0]

    if m != n and flag_simple_U:
        raise ValueError('If m is not equal to n, flag_simple_U must be False')

    samples1_exp1 = torch.unsqueeze(samples1, dim=1)  # N x 1 x dim
    samples1_exp2 = torch.unsqueeze(samples1, dim=0)  # 1 x N x dim

    samples2_exp1 = torch.unsqueeze(samples2, dim=1)  # N x 1 x dim
    samples2_exp2 = torch.unsqueeze(samples2, dim=0)  # 1 x N x dim

    # Term1
    K1 = kernel(samples1_exp1, samples1_exp2, kernel_hyper=kernel_hyper)  # N x N
    if flag_U:
        K1 = K1 - torch.diag(torch.diag(K1))
    # Term3
    K3 = kernel(samples2_exp1, samples2_exp2, kernel_hyper=kernel_hyper)  # N x N
    if flag_U:
        K3 = K3 - torch.diag(torch.diag(K3))

    # Term2
    if flag_simple_U:
        K2_comp = kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper)
        K2_comp = K2_comp - torch.diag(torch.diag(K2_comp))
        K2 = K2_comp + K2_comp.t()
    else:
        K2 = 2 * kernel(samples1_exp1, samples2_exp2, kernel_hyper=kernel_hyper)  # N x N

    if flag_U:
        if flag_simple_U:
            MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * (m - 1)) * torch.sum(K2)

        else:
            MMD = torch.sum(K1) / (n * (n - 1)) + torch.sum(K3) / (m * (m - 1)) - 1. / (m * n) * torch.sum(K2)
    else:
        MMD = torch.sum(K1 + K3 - K2) / (m * n)

    return MMD, K1 + K3 - K2


def bootstrap_MMD(n_boots, p_samples, q_samples, kernel, kernel_hyper):
    n_sample = q_samples.shape[0]

    weights = np.random.multinomial(n_sample, np.ones(n_sample) / n_sample, size=int(n_boots))
    weights = weights / n_sample

    weights = torch.from_numpy(weights).type(q_samples.type())

    MMD, MMD_comp = compute_MMD(p_samples, q_samples, kernel, kernel_hyper, flag_U=True, flag_simple_U=True)

    # Now compute bootstrap samples
    # now compute boots strap samples
    with torch.no_grad():
        MMD_comp = torch.unsqueeze(MMD_comp, dim=0)  # 1 x N x N
        weights_exp = torch.unsqueeze(weights, dim=-1)  # m x N x 1
        weights_exp2 = torch.unsqueeze(weights, dim=1)  # m x 1 x n

        bootstrap_samples = (weights_exp - 1. / n_sample) * MMD_comp * (weights_exp2 - 1. / n_sample)  # m x N x N

        bootstrap_samples = torch.sum(torch.sum(bootstrap_samples, dim=-1), dim=-1)

    return MMD, bootstrap_samples
