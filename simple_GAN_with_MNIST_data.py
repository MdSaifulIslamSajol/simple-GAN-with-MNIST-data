# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 09:10:31 2022

@author: msajol1
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
#%%

class Discriminator(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # normalize inputs to [-1, 1] so make outputs [-1, 1]
        )

    def forward(self, x):
        return self.gen(x)

#%%
# Hyperparameters etc.
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64  # noise 
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 50
#%%
# creating instance of discriminator and generator
disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_randn_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),]
)

dataset = datasets.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# optimizer
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)

# loss function
criterion = nn.BCELoss()

# tensor board
writer_fake_images = SummaryWriter(f"logs/fake_images")
writer_real_images = SummaryWriter(f"logs/real_images")
step = 0

#%%
for epoch in range(num_epochs):
    
    for batch_idx, (real_images, _) in enumerate(loader):
        
        real_images = real_images.view(-1, 784).to(device)  # no labels needed
        batch_size = real_images.shape[0]

        ### Train Discriminator Network: max log(D(x)) + log(1 - D(G(z)))
        randn_noise = torch.randn(batch_size, z_dim).to(device)
        fake_images = gen(randn_noise)
        
        disc_real_images = disc(real_images).view(-1)
        lossD_real_images = criterion(disc_real_images, torch.ones_like(disc_real_images))
        
        disc_fake_images = disc(fake_images).view(-1)
        lossD_fake_images = criterion(disc_fake_images, torch.zeros_like(disc_fake_images))
        
        lossD = (lossD_real_images + lossD_fake_images) / 2  # loss discriminator
        
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator Network: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake_images).view(-1)
        lossG = criterion(output, torch.ones_like(output))  # loss generator
        
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake_images = gen(fixed_randn_noise).reshape(-1, 1, 28, 28)
                data = real_images.reshape(-1, 1, 28, 28)
                img_grid_fake_images = torchvision.utils.make_grid(fake_images, normalize=True)
                img_grid_real_images = torchvision.utils.make_grid(data, normalize=True)

                writer_fake_images.add_image(
                    "Mnist fake_images Images", img_grid_fake_images, global_step=step
                )
                writer_real_images.add_image(
                    "Mnist real_images Images", img_grid_real_images, global_step=step
                )
                step += 1