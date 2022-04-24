import torch
import torch.nn.functional as F


class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()

        def generator_block(in_filters, out_filters):
            block = [
                torch.nn.ConvTranspose2d(in_filters, out_filters, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(out_filters, 0.8),
                torch.nn.LeakyReLU(0.2, inplace=True)
            ]
            return block

        self.conditional, input_size = args.conditional, args.z_size
        if self.conditional:
            self.label_emb = torch.nn.Embedding(args.n_classes, args.n_classes)
            input_size = input_size + args.n_classes

        self.init_size = args.img_size // 4
        self.first = torch.nn.Sequential(torch.nn.Linear(input_size, 128 * self.init_size ** 2))

        self.conv_blocks = torch.nn.Sequential(
            *generator_block(128, 128),
            *generator_block(128, args.img_size),
            *generator_block(args.img_size, args.img_size // 2),
            torch.nn.ConvTranspose2d(args.img_size // 2, args.img_size // 4, 4, 2, 1),
            torch.nn.BatchNorm2d(args.img_size // 4, 0.8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.ConvTranspose2d(args.img_size // 4, args.channel, 4, 2, 1),
            torch.nn.Tanh()
        )

    def forward(self, z, label):
        z = torch.cat((self.label_emb(label), z), -1) if self.conditional else z
        z = self.first(z)
        z = z.view(z.shape[0], 128, self.init_size, self.init_size)
        out = self.conv_blocks(z)  # [bs, channel, img_size, img_size])
        return out


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        sn_fn = torch.nn.utils.spectral_norm if not args.disable_sn else lambda x: x

        def discriminator_block(in_filters, out_filters):
            block = [
                sn_fn(torch.nn.Conv2d(in_filters, out_filters, 3, 2, 1)),
                torch.nn.LeakyReLU(0.2, inplace=True)
            ]
            return block

        ds_size = args.img_size // 2 ** 4
        self.conditional, input_size = args.conditional, args.channel
        if self.conditional:
            self.label_emb = torch.nn.Embedding(args.n_classes, args.img_size ** 2)
            input_size = input_size + 1

        self.model = torch.nn.Sequential(
            *discriminator_block(input_size, 16),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128)
        )

        # The height and width of downsampled image
        self.adv_layer = torch.nn.Sequential(sn_fn(torch.nn.Linear(128 * ds_size ** 2, args.latent_dim)))

    def forward(self, x, label):
        if self.conditional:
            label = self.label_emb(label).view(x.shape[0], 1, x.shape[2], x.shape[3])  # [bs, 1, img_size. img_size]
            x = torch.cat((label, x), dim=1)  # [bs, channel+1, img_size. img_size]

        out = self.model(x)
        out = out.view(out.shape[0], -1)  # [bs, 512]
        validity = self.adv_layer(out)  # [bs, latent_dim]
        return validity


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def hinge_loss_dis(fake, real):
    assert fake.dim() == 2 and fake.shape[1] == 1 and real.shape == fake.shape
    loss = F.relu(1.0 - real).mean() + F.relu(1.0 + fake).mean()
    return loss


def hinge_loss_gen(fake):
    assert fake.dim() == 2 and fake.shape[1] == 1
    loss = -fake.mean()
    return loss
