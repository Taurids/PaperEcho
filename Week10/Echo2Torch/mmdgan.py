from torch import nn


class Generator(nn.Module):
    def __init__(self, z_size, channels):
        super(Generator, self).__init__()
        self.init_size = z_size * 2
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(z_size, self.init_size, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2)
        )

        def decoder_block(in_filters, out_filters):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_filters, 0.8),
                nn.LeakyReLU(0.2)
            ]
            return block

        self.conv_blocks = nn.Sequential(
            *decoder_block(self.init_size, self.init_size // 2),
            *decoder_block(self.init_size // 2, self.init_size // 4),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(self.init_size // 4, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.unsqueeze(2).unsqueeze(3)  # [bs, 100, 1, 1]
        out = self.l1(z)  # [bs, 64, 4, 4]

        out = self.conv_blocks(out)  # [bs, 16, 16, 16]
        return self.last(out)  # [bs, 1, 32, 32]


class Discriminator(nn.Module):
    def __init__(self, z_size, channels, sn=False):
        super(Discriminator, self).__init__()
        sn_fn = nn.utils.spectral_norm if sn else lambda x: x

        def encoder_block(in_filters, out_filters, bn=True):
            block = [sn_fn(nn.Conv2d(in_filters, out_filters, 4, 2, 1)), nn.ReLU()]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(channels, 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            sn_fn(nn.Conv2d(64, z_size, 4, 1, 0, bias=False))
        )

    def forward(self, x):
        out = self.model(x)  # [bs, 100, 1, 1]
        return out.view(out.shape[0], -1)  # [bs, 100]


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
