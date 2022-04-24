import os
import argparse
import numpy as np
import torch
import Echo2Torch
import torchvision
from torch.autograd import Variable

path_dir = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
parser.add_argument("--z_size", type=int, default=128, help="dimensionality of the noise as generator input")
parser.add_argument("--latent_dim", type=int, default=1, help="dimensionality of the latent space")
parser.add_argument('--disable_sn', default=False, action='store_true')
parser.add_argument('--conditional', default=True, action='store_true')
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channel", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=3000, help="interval between image sampling")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--warmup_proportion", type=float, default=0.9,
                    help="Proportion of training to perform linear learning rate warmup for. "
                         "E.g., 0.1 = 10%% of training.")
parser.add_argument('--dir_dataset', type=str, default=os.path.join(path_dir, "./datasets/"))
parser.add_argument('-scaling_coeff', type=float, default=10., help='coeff of scaling [%(default)s]')
args = parser.parse_args()
print(args)

# Configure data loader
os.makedirs(args.dir_dataset, exist_ok=True)
ds_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(args.img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
ds_instance = torchvision.datasets.CIFAR10(args.dir_dataset, train=True, download=True, transform=ds_transform)
train_loader = torch.utils.data.DataLoader(ds_instance, batch_size=args.batch_size, drop_last=True, shuffle=True)

# reinterpret command line inputs
torch.manual_seed(77)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageMetric = Echo2Torch.metrics.Image(args, _isc=True, _fid=True, _kid=True)

# create Generator and Discriminator models
net_G = Echo2Torch.mmdgan.Generator(args).to(device).train()
net_D = Echo2Torch.mmdgan.Discriminator(args).to(device).train()

# Initialize weights
net_G.apply(Echo2Torch.mmdgan.weights_init_normal)
net_D.apply(Echo2Torch.mmdgan.weights_init_normal)

# Optimizer
optimizer_G = torch.optim.Adam(net_G.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(net_D.parameters(), lr=args.lr, betas=(0.5, 0.999))
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=args.warmup_proportion)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=30, gamma=args.warmup_proportion)

# Prepare model
output_dir = os.path.join(path_dir, "output_dir")
os.makedirs(os.path.join(path_dir, "output_dir"), exist_ok=True)
if os.path.exists(os.path.join(output_dir, "netG.pth")):
    net_G.load_state_dict(torch.load(os.path.join(output_dir, "netG.pth"), map_location='cpu'))
    net_D.load_state_dict(torch.load(os.path.join(output_dir, "netD.pth"), map_location='cpu'))
    optimizer_G.load_state_dict(torch.load(os.path.join(output_dir, "optimG.bin"), map_location='cpu'))
    optimizer_D.load_state_dict(torch.load(os.path.join(output_dir, "optimD.bin"), map_location='cpu'))
    scheduler_G.load_state_dict(torch.load(os.path.join(output_dir, "schedG.bin"), map_location='cpu'))
    scheduler_D.load_state_dict(torch.load(os.path.join(output_dir, "schedD.bin"), map_location='cpu'))


# ----------
#  Training
# ----------
torch.cuda.empty_cache()
for epoch in range(args.n_epochs):
    for i, (img, label) in enumerate(train_loader):
        real_img = Variable(img.type(torch.FloatTensor)).to(device)
        label = Variable(label.type(torch.LongTensor)).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        net_D.requires_grad_(True)
        net_G.requires_grad_(False)

        # Sample noise and labels as generator input
        z = torch.randn(args.batch_size, args.z_size, device=device)

        # zero the parameter gradients
        optimizer_D.zero_grad()

        # Generate a batch of images
        fake_img = net_G(z, label).detach()

        # Measure discriminator's ability to classify real from generated samples
        Enc_X_D, Enc_Y_D = net_D(real_img, label), net_D(fake_img, label)
        Enc_X_D, Enc_Y_D = torch.mean(Enc_X_D, dim=-1, keepdim=True), torch.mean(Enc_Y_D, dim=-1, keepdim=True)
        hinge_lossD = Echo2Torch.mmdgan.hinge_loss_dis(Enc_Y_D, Enc_X_D)
        loss_D = hinge_lossD

        loss_D.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        # Train the generator every n_critic iterations
        if (i + 1) % args.n_critic == 0:
            net_D.requires_grad_(False)
            net_G.requires_grad_(True)
            # zero the parameter gradients
            optimizer_G.zero_grad()
            # Generate a batch of images
            fake_img = net_G(z, label)
            # Loss measures generator's ability to fool the discriminator
            Enc_Y_D = torch.mean(net_D(fake_img, label), dim=-1, keepdim=True)
            hinge_lossG = Echo2Torch.mmdgan.hinge_loss_gen(Enc_Y_D)
            loss_G = hinge_lossG

            loss_G.backward()
            optimizer_G.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.n_epochs, i, len(train_loader), loss_D.item(), loss_G.item())
            )

        batches_done = epoch * len(train_loader) + i
        if batches_done % args.sample_interval == 0:
            MetricDict = ImageMetric(real_img, fake_img)
            isc_score, fid_score,  kid_score = MetricDict['ISC'], MetricDict['FID'], MetricDict['KID']
            print(
                    "Metric --> [ISC: %f+%f] [FID: %f] [KID: %f+%f]"
                    % (isc_score[0], isc_score[1], fid_score[0], kid_score[0], kid_score[1])
            )
            # initialize persistent noise for observed samples
            n_row = args.n_classes
            z_vis = torch.randn(n_row ** 2, args.z_size, device=device)
            z_vis = Variable(z_vis, requires_grad=False)
            """Saves a grid of generated digits ranging from 0 to n_classes"""
            # Get labels ranging from 0 to n_classes for n rows
            gen_label = np.array([num for num in range(n_row) for _ in range(n_row)])
            gen_label = torch.autograd.Variable(torch.LongTensor(gen_label)).to(device)
            gen_img = net_G(z_vis, gen_label)
            torchvision.utils.save_image(gen_img.data, "images/%d.png" % batches_done, nrow=n_row)
            torch.save(net_G.state_dict(), os.path.join(output_dir, 'netG.pth'))
            torch.save(net_D.state_dict(), os.path.join(output_dir, 'netD.pth'))
            torch.save(optimizer_G.state_dict(), os.path.join(output_dir, 'optimG.bin'))
            torch.save(optimizer_D.state_dict(), os.path.join(output_dir, 'optimD.bin'))
            torch.save(scheduler_G.state_dict(), os.path.join(output_dir, 'schedG.bin'))
            torch.save(scheduler_D.state_dict(), os.path.join(output_dir, 'schedD.bin'))
    # decay LR
    scheduler_G.step()
    scheduler_D.step()
