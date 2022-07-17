import argparse
import time
from src.utils import median_heruistic
from src.gof_test import bootstrap_KSD, bootstrap_MMD
from src.SE_kernel import SE_kernel_multi, trace_SE_kernel_multi
import numpy as np
from tqdm import tqdm
import torch

from src.distribution import MultivariateLaplace, MultivariateT

# Arguments for scripts
parser = argparse.ArgumentParser(description='Benchmark GOF test')
parser.add_argument('--n_trials', type=int, default=100, help="number of GOF trials")
parser.add_argument('--n_samples', type=int, default=1000, help="number of total sample used")
parser.add_argument('--method', type=str, default='KSD', help="GOF test method, either KSD or MMD")
parser.add_argument('--distribution', type=str, default='diffusion',
                    help="test distributions: Laplace, diffusion, multi-t or Null")

args = parser.parse_args()
dim = 7
# number of bootstrap samples, cause large GPU memory usage. Reduce this if CUDA out of memory.
n_boots = 100
# significance level for GOF
significance = 0.05

# For storing results
KSD_results = np.zeros(args.n_trials)
MMD_results = np.zeros(args.n_trials)

KSD_value = np.zeros(args.n_trials)
MMD_value = np.zeros(args.n_trials)
# For storing time usage
time_grad = 0
time_KSD_avg = 0
# For storing Correct rejection
KSD_correct = np.zeros(args.n_trials)
MMD_correct = np.zeros(args.n_trials)

# Set distribution p
p_cov = torch.eye(dim)
p_mean = torch.zeros(dim)
sample_size_full = torch.Size([args.n_samples])  # total sample number
p_x = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, p_cov)
# Set distribution q
if args.distribution == 'Laplace':
    sample_size = torch.Size([args.n_samples])
    q_cov = 1. / np.sqrt(2) * torch.ones(dim)
    H_1_q = MultivariateLaplace(p_mean, q_cov)
elif args.distribution == 'multi-t':
    p_cov = p_cov * 5 / (5 - 2.0)
    sample_size = torch.Size([args.n_samples, dim])
    H_1_q = MultivariateT(df=5, loc=0)
elif args.distribution == 'diffusion':
    q_cov = p_cov.clone().detach()
    q_cov[0, 0] = 0.3
    sample_size = torch.Size([args.n_samples])
    H_1_q = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, q_cov)
else:
    sample_size = torch.Size([args.n_samples])
    H_1_q = torch.distributions.multivariate_normal.MultivariateNormal(p_mean, p_cov)

# Start the trials
for idx_trial in tqdm(range(args.n_trials)):
    # Draw samples
    p_samples = p_x.rsample(sample_size_full)
    q_samples = H_1_q.rsample(sample_size_full).requires_grad_()

    test_sample = q_samples.clone().detach().requires_grad_()

    # Start GOF Test
    if args.method == 'KSD':
        # ——---------------- KSD Test --------------------------
        start_time = time.time()
        median_dist = median_heruistic(q_samples, q_samples.clone())
        bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
        kernel_hyper_KSD = {
            'bandwidth': bandwidth
        }
        KSD, KSD_bootstrap = bootstrap_KSD(
            n_boots, p_x, q_samples.clone().requires_grad_(),
            SE_kernel_multi, trace_SE_kernel_multi,
            kernel_hyper=kernel_hyper_KSD
        )
        prop_KSD = torch.mean((KSD_bootstrap >= KSD).float().cuda())
        time_KSD = time.time() - start_time
        time_KSD_avg += time_KSD

        KSD_value[idx_trial] = KSD.cpu().data.numpy()

        if prop_KSD < significance:
            # reject alternative
            KSD_correct[idx_trial] = 1
        else:
            KSD_correct[idx_trial] = 0
        if (idx_trial + 1) % 1 == 0:
            print('trial:%s test_acc:%s prop_KSD:%s KSD:%s time:%s' % (
                idx_trial, KSD_correct[0:idx_trial + 1].sum() / (idx_trial + 1), prop_KSD,
                KSD.cpu().data.numpy(), time_KSD_avg / (idx_trial + 1)))
    elif args.method == "MMD":
        # ——---------------- MMD Test --------------------------
        agg_sample = torch.cat((q_samples, p_samples), dim=0)
        median_dist = median_heruistic(agg_sample, agg_sample.clone())
        bandwidth = 2 * torch.pow(0.5 * median_dist, 0.5)
        kernel_hyper_MMD = {
            'bandwidth': bandwidth
        }
        MMD, MMD_bootstrap = bootstrap_MMD(n_boots, p_samples, q_samples, SE_kernel_multi,
                                           kernel_hyper=kernel_hyper_MMD
                                           )
        prop_MMD = torch.mean((MMD_bootstrap >= MMD).float().cuda())
        MMD_value[idx_trial] = MMD.cpu().data.numpy()

        if prop_MMD < significance:
            # reject alternative
            MMD_correct[idx_trial] = 1.
        else:
            MMD_correct[idx_trial] = 0.
        if (idx_trial + 1) % 1 == 0:
            print('trial:%s test_acc:%s prop_MMD:%s MMD:%s' % (
                idx_trial, MMD_correct[0:idx_trial + 1].sum() / (idx_trial + 1), prop_MMD,
                MMD.cpu().data.numpy()))

# Finished all trials and process the results
KSD_test = np.sum(KSD_correct) / args.n_trials
MMD_test = np.sum(MMD_correct) / args.n_trials

KSD_v = np.mean(KSD_value)
MMD_v = np.mean(MMD_value)

KSD_results = KSD_correct
MMD_results = MMD_correct

if args.method == 'KSD':
    print('KSD_result:%s KSD_value:%s KSD_time:%s' % (KSD_test, KSD_v, time_KSD_avg / args.n_trials))
elif args.method == 'MMD':
    print('MMD_result:%s MMD_value:%s' % (MMD_test, MMD_v))
