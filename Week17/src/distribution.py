import torch
import numpy as np


class MultivariateLaplace(object):
    def __init__(self, loc, scale):
        self.comp = torch.distributions.laplace.Laplace(loc, scale)
        self.dim = loc.shape[0]

    def rsample(self, size):
        return self.comp.rsample(size)

    def log_prob(self, X):
        log_prob = torch.sum(self.comp.log_prob(X), dim=-1)
        return log_prob

    def sample(self, size):
        return self.rsample(torch.Size([size]))


class MultivariateT(object):
    def __init__(self, loc, df, scale=1):
        self.scale = scale
        self.loc = loc
        self.df = df

        # Create univariate student t distribution
        self.comp_dist = torch.distributions.studentT.StudentT(self.df, loc=self.loc, scale=self.scale)

    def rsample(self, size):
        # size should be N x D
        return self.comp_dist.rsample(size)

    def log_prob(self, X):
        log_prob = torch.sum(self.comp_dist.log_prob(X), dim=-1)
        return log_prob

    def sample(self, size):
        return self.rsample(torch.Size([size]))
