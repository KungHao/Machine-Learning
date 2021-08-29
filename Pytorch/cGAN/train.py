import os
import torch
import logging
import numpy as np
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image

from utils import *
from model import Generator, Discriminator

logger = set_logger("cGAN")

# Hyper parameters
EPOCH = 1000
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001

keep_train = 0
G_losses = []
D_losses = []
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.ToTensor()
data_train = datasets.MNIST(root="../dataset",
                            transform=transform,
                            train=True,
                            download=True)
data_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)

D = Discriminator().to(device)
G = Generator().to(device)
criterion = nn.MSELoss().to(device)
D_optim = torch.optim.Adam(D.parameters(), lr=LR_D)
G_optim = torch.optim.Adam(G.parameters(), lr=LR_G)

sample_z = torch.randn(10, 100).to(device)
sample_label = torch.arange(10).to(device)

if keep_train != 0:
    G, D = load_checkpoint(G, D, keep_train)

for epoch in range(EPOCH+keep_train):
    print('start epoch {}...'.format(epoch))
    for i, (img, label) in enumerate(data_loader):
        
        batch_size =img.shape[0]
        # batch_size = BATCH_SIZE
        
        z = torch.FloatTensor(torch.randn(batch_size, 100)).to(device)
        
#         real_label = label.view(label.size(0), 0)
        real_label = torch.LongTensor(label).to(device)
        
        fake_label = torch.LongTensor(torch.randint(0, 10, (batch_size,))).to(device)

#         fake_label = label.view(label.size(0), -1)
        
        real_img = img.view(batch_size, -1).to(device)
        fake_img = G(z, fake_label).to(device)
        
        ones = torch.ones(batch_size, 1).to(device)
        zeros = torch.zeros(batch_size, 1).to(device)
        
        D_real = D(real_img, real_label)
        D_fake = D(fake_img, fake_label)

#         D_input = torch.cat((prob_real, prob_fake), 0)
#         D_target = torch.cat((real_label, fake_label), 0)

        D_loss_real = criterion(D_real, ones)      
        D_loss_fake = criterion(D_fake, zeros)
        D_loss = (D_loss_real + D_loss_fake) / 2
        G_loss = criterion(D_fake, ones)

        D_optim.zero_grad()
        D_loss.backward(retain_graph=True)      # reusing computational graph
        D_optim.step()

        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
        
        if i % 100 == 0 or i == len(data_loader):
            G_losses.append(G_loss.item())
            D_losses.append(D_loss.item())
            print('{}/{}, {}/{}, D_loss: {:.3f}  G_loss: {:.3f}'.format(epoch+1, EPOCH, i, len(data_loader), D_loss.item(), G_loss.item()))
            log = '{}/{}, {}/{}, D_loss: {:.3f}  G_loss: {:.3f}'.format(epoch+1, EPOCH, i, len(data_loader), D_loss.item(), G_loss.item())
            logger.info(log)

    if (epoch+1) % 50 == 0:
        sample_img = G(sample_z, sample_label)
        save_image(reshape(sample_img).data.cpu(),
                    os.path.join('./result', 'sample_{}.png'.format(epoch+1)),
                    nrow=1,
                    padding=2)
        torch.save(G.state_dict(),
                    os.path.join('./checkpoint', 'G_{}.pth'.format(epoch+1)))
        torch.save(D.state_dict(),
                    os.path.join('./checkpoint', 'D_{}.pth'.format(epoch+1)))
plt_loss(G_losses, D_losses)
print('Done!!!')