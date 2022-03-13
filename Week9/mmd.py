import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_mmd2(f_of_X, f_of_Y):
    delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
    loss = delta.dot(delta.T)
    return loss


def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size()[0]), int(total.size()[0]), int(total.size()[1])
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size()[0]), int(total.size()[0]), int(total.size()[1])
    )
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list*2]
    return sum(kernel_val)


def rational_quadratic_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_gamma=1):
    # n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size()[0]), int(total.size()[0]), int(total.size()[1])
    )
    total1 = total.unsqueeze(1).expand(
        int(total.size()[0]), int(total.size()[0]), int(total.size()[1])
    )
    L2_distance = ((total0 - total1) ** 2).sum(2)
    bandwidth = fix_gamma
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    kernel_val = [torch.pow(1 + L2_distance / (2*bandwidth_temp), -bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None, fix_gamma=1):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.fix_gamma = fix_gamma
        self.kernel_type = kernel_type

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = gaussian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma
            )
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = F.relu(torch.mean(XX + YY - XY - YX))
            return loss
        elif self.kernel_type == 'rq':
            batch_size = int(source.size()[0])
            kernels = rational_quadratic_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_gamma=self.fix_gamma
            )
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = F.relu(torch.mean(XX + YY - XY - YX))
            return loss
