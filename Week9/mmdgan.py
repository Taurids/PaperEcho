import argparse
import os
import numpy as np

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import mmd

os.makedirs("images", exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.1, help="lower and upper clip value for disc. weights")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init_normal(m):
    class_name = m.__class__.__name__
    if class_name.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# input: bs * latent_dim
# output: bs * channels * img_size * img_size
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.init_size = opt.img_size * 2
        self.l1 = nn.Sequential(
            nn.ConvTranspose2d(opt.latent_dim, self.init_size, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

        def decoder_block(in_filters, out_filters):
            block = [
                nn.ConvTranspose2d(in_filters, out_filters, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_filters, 0.8),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return block

        self.conv_blocks = nn.Sequential(
            *decoder_block(self.init_size, self.init_size//2),
            *decoder_block(self.init_size//2, self.init_size//4),
        )

        self.last = nn.Sequential(
            nn.ConvTranspose2d(self.init_size//4, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, De_z):
        De_z = De_z.unsqueeze(2).unsqueeze(3)  # [bs, 100, 1, 1]
        out = self.l1(De_z)  # [bs, 128, 4, 4]

        out = self.conv_blocks(out)  # [bs, 8, 32, 32]
        img = self.last(out)  # [bs, 1, 64, 64]
        return img


# input: bs * channels * img_size * img_size
# output: bs * latent_dim
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        def encoder_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *encoder_block(opt.channels, 16, bn=False),
            *encoder_block(16, 32),
            *encoder_block(32, 64),
            nn.Conv2d(64, opt.latent_dim, 4, 1, 0, bias=False)
        )

    def forward(self, img):
        out = self.model(img)  # [bs, 100, 1, 1]
        out = out.view(out.shape[0], -1)  # [bs, 100]
        return out


# Generator is a decoder
class Generator(nn.Module):
    def __init__(self, decoder):
        super(Generator, self).__init__()
        self.decoder = decoder

    def forward(self, x):
        return self.decoder(x)


# Discriminator is an encoder + decoder
class Discriminator(nn.Module):
    def __init__(self, encoder, decoder):
        super(Discriminator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        enc_X = self.encoder(x)
        dec_X = self.decoder(enc_X)
        return enc_X, dec_X


class OneSided(nn.Module):
    def __init__(self):
        super(OneSided, self).__init__()
        main = nn.ReLU()
        self.main = main

    def forward(self, x):
        output = self.main(-x)
        output = -output.mean()
        return output


# Loss function
# adversarial_loss = torch.nn.BCELoss()
one_sided = OneSided()
# mmd_loss = mmd.MMDLoss(kernel_type='rbf', fix_sigma=4)  # 1, 2, 4, 8, 16
mmd_loss = mmd.MMDLoss(kernel_type='rq')  # 0.25, 0.5, 1, 2, 4


# Initialize generator and discriminator
G_decoder = Decoder()
D_encoder, D_decoder = Encoder(), Decoder()
generator = Generator(G_decoder)
discriminator = Discriminator(D_encoder, D_decoder)

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("./data/mnist/", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/mnist/",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr)

generator.to(device)
discriminator.to(device)
mmd_loss.to(device)
one_sided.to(device)

# generator.load_state_dict(torch.load('netG_iter.pth', map_location='cpu'))
# discriminator.load_state_dict(torch.load('netD_iter.pth', map_location='cpu'))

# ----------
#  Training
# ----------
gen_iterations = 0
fixed_noise = torch.FloatTensor(64, opt.latent_dim).normal_(0, 1).to(device)
fixed_noise = Variable(fixed_noise, requires_grad=False)
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        # Configure input
        real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for p in discriminator.parameters():
            p.requires_grad = True
        for q in generator.parameters():
            q.requires_grad = False

        optimizer_D.zero_grad()

        # Sample noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
        # Generate a batch of images
        fake_imgs = generator(z).detach()

        # Measure discriminator's ability to classify real from generated samples
        f_enc_X_D, f_dec_X_D = discriminator(real_imgs)
        f_enc_Y_D, f_dec_Y_D = discriminator(fake_imgs)

        # compute rank hinge loss
        one_side_errD = one_sided(f_enc_X_D.mean(0) - f_enc_Y_D.mean(0))
        # compute L2-loss of AE
        L2_AE_X_D = torch.mean(torch.pow(real_imgs - f_dec_X_D, 2))
        L2_AE_Y_D = torch.mean(torch.pow(fake_imgs - f_dec_Y_D, 2))
        # compute mmd loss
        mmd_errD = mmd_loss(f_enc_X_D, f_enc_Y_D)
        # compute the final loss
        loss_D = mmd_errD + 0.9 * one_side_errD - 0.7 * L2_AE_X_D - 0.7 * L2_AE_Y_D

        d_loss = -1 * loss_D
        d_loss.backward()
        optimizer_D.step()

        for p in discriminator.encoder.parameters():
            p.data.clamp_(-opt.clip_value, opt.clip_value)

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            for p in discriminator.parameters():
                p.requires_grad = False
            for q in generator.parameters():
                q.requires_grad = True

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            f_enc_X, f_dec_X = discriminator(real_imgs)
            f_enc_Y, f_dec_Y = discriminator(gen_imgs)

            # compute rank hinge loss
            one_side_errG = one_sided(f_enc_X.mean(0) - f_enc_Y.mean(0))
            # compute mmd loss
            mmd_errG = mmd_loss(f_enc_X, f_enc_Y)
            # compute the final loss
            g_loss = mmd_errG + 0.9 * one_side_errG

            g_loss.backward()
            optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), loss_D.item(), g_loss.item())
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            gen_imgs = generator(fixed_noise)
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
            torch.save(generator.state_dict(), 'netG_iter.pth')
            torch.save(discriminator.state_dict(), 'netD_iter.pth')
