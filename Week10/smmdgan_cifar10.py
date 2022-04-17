import os
import argparse
import torch
from Echo2Torch import mmd, mmdgan
from torchvision import transforms, datasets, utils
from torch.autograd import Variable

data_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=37, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0005, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument('--disable_sn', default=True, action='store_true')
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=1, help="lower and upper clip value for disc. weights")
parser.add_argument("--warmup_proportion", type=float, default=0.9,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--dir_dataset', type=str, default='./data/cifar10/')
parser.add_argument('-scaling_coeff', type=float, default=10., help='coeff of scaling [%(default)s]')
args = parser.parse_args()
print(args)

# Configure data loader
os.makedirs(args.dir_dataset, exist_ok=True)
ds_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
ds_instance = datasets.CIFAR10(args.dir_dataset, train=True, download=True, transform=ds_transform)
# ds_instance = datasets.MNIST(args.dir_dataset, train=True, download=True, transform=ds_transform)
loader_iter = torch.utils.data.DataLoader(ds_instance, batch_size=args.batch_size, drop_last=True, shuffle=True)

# reinterpret command line inputs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create Generator and Discriminator models
net_G = mmdgan.Generator(args.latent_dim, args.channels).to(device).train()
net_D = mmdgan.Discriminator(args.latent_dim, args.channels, not args.disable_sn).to(device).train()

# initialize persistent noise for observed samples
z_vis = torch.FloatTensor(16, args.latent_dim).normal_(0, 1).to(device)
fixed_noise = Variable(z_vis, requires_grad=False)


# Initialize weights
net_G.apply(mmdgan.weights_init_normal)
net_D.apply(mmdgan.weights_init_normal)

optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr)
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr)
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=args.warmup_proportion)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=args.warmup_proportion)

# Loss function
mmd_loss = mmd.MMDLoss(kernel_type='rq').to(device)  # 0.25, 0.5, 1, 2, 4
# mmd_loss = mmd.MMDLoss(kernel_type='rbf', fix_sigma=4).to(device)  # 0.25, 0.5, 1, 2, 4

# ----------
#  Training
# ----------
torch.cuda.empty_cache()
for epoch in range(args.n_epochs):
    for i, (img, _) in enumerate(loader_iter):
        real_img = Variable(img.type(torch.FloatTensor)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        net_D.requires_grad_(True)
        net_G.requires_grad_(False)
        real_img.requires_grad_(True)
        # Sample noise as generator input
        z = Variable(torch.FloatTensor(img.shape[0], args.latent_dim).normal_(0, 1).to(device))
        # zero the parameter gradients
        optimizer_D.zero_grad()
        # Generate a batch of images
        fake_img = net_G(z).detach()
        # Measure discriminator's ability to classify real from generated samples
        Enc_X_D, Enc_Y_D = net_D(real_img), net_D(fake_img)
        # compute scaled-mmd loss
        mmd_errD = mmd_loss(Enc_X_D, Enc_Y_D)
        scale = mmd.calculate_scaled(Enc_X_D, real_img, args.scaling_coeff)
        loss_D = mmd_errD * scale

        d_loss = -1 * loss_D
        d_loss.backward()
        # torch.nn.utils.clip_grad_norm_(net_D.parameters(), max_norm=1)
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        # Train the generator every n_critic iterations
        if (i + 1) % args.n_critic == 0:
            net_D.requires_grad_(False)
            net_G.requires_grad_(True)
            real_img.requires_grad_(False)
            # zero the parameter gradients
            optimizer_G.zero_grad()
            # Generate a batch of images
            fake_img = net_G(z)
            # Loss measures generator's ability to fool the discriminator
            mmd_errG = mmd_loss(net_D(real_img), net_D(fake_img))
            loss_G = mmd_errG

            g_loss = loss_G
            g_loss.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(loader_iter), loss_D.item(), loss_G.item())
            )

        batches_done = epoch * len(loader_iter) + i
        if batches_done % args.sample_interval == 0:
            gen_img = net_G(z_vis)
            utils.save_image(gen_img.data[:9], "images/%d.png" % batches_done, nrow=3, normalize=True)
            torch.save(net_G.state_dict(), 'netG_iter.pth')
            torch.save(net_D.state_dict(), 'netD_iter.pth')
    # decay LR
    scheduler_G.step()
    scheduler_D.step()
