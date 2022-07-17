import torch


def median_heruistic(sample1, sample2):
    with torch.no_grad():
        G = torch.sum(sample1 * sample1, dim=-1)  # N or * x N
        G_exp = torch.unsqueeze(G, dim=-2)  # 1 x N or * x1 x N

        H = torch.sum(sample2 * sample2, dim=-1)
        H_exp = torch.unsqueeze(H, dim=-1)  # N x 1 or * * x N x 1
        dist = G_exp + H_exp - 2 * sample2.matmul(torch.transpose(sample1, -1, -2))  # N x N or  * x N x N
        if len(dist.shape) == 3:
            dist = dist[torch.triu(torch.ones(dist.shape)) == 1].view(dist.shape[0], -1)  # * x (NN)
            median_dist, _ = torch.median(dist, dim=-1)  # *
        else:
            dist = (dist - torch.tril(dist)).view(-1)
            median_dist = torch.median(dist[dist > 0.])
    return median_dist.clone().detach()
