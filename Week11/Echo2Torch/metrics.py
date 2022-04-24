import torch
from torchmetrics.image import inception, fid, kid


class Image(torch.nn.Module):
    def __init__(self, args, _isc=False, _fid=False, _kid=False):
        super(Image, self).__init__()
        # Prepare metric
        self._isc, self._fid, self._kid = _isc, _fid, _kid
        self.ImageMetricDict = {'ISC': [0, 0], 'FID': [0, 0], 'KID': [0, 0]}
        self.isc_metric = inception.InceptionScore() if self._isc else None
        self.fid_metric = fid.FrechetInceptionDistance(feature=64) if self._fid else None
        self.kid_metric = kid.KernelInceptionDistance(subset_size=args.batch_size // 2) if self._kid else None

    def forward(self, real, fake):
        # Prepare data
        real = (255 * (real.clamp(-1, 1) * 0.5 + 0.5))
        real = real.cpu().to(torch.uint8)
        fake = (255 * (fake.clamp(-1, 1) * 0.5 + 0.5))
        fake = fake.cpu().to(torch.uint8)
        # Inception Score
        if self._isc:
            self.isc_metric.update(fake)
            isc_mean, isc_std = self.isc_metric.compute()
            self.ImageMetricDict['ISC'] = [isc_mean.item(), isc_std.item()]
        # Frechet Inception Distance
        if self._fid:
            self.fid_metric.update(real, real=True)
            self.fid_metric.update(fake, real=False)
            fid_score = self.fid_metric.compute()
            self.ImageMetricDict['FID'] = [fid_score.item(), 0]
        # Kernel Inception Distance
        if self._kid:
            self.kid_metric.update(real, real=True)
            self.kid_metric.update(fake, real=False)
            kid_mean, kid_std = self.kid_metric.compute()
            self.ImageMetricDict['KID'] = [kid_mean.item(), kid_std.item()]
        return self.ImageMetricDict


