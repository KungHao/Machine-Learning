import os
import torch
import logging
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.utils import save_image

from utils import *
from model import Generator, Discriminator

logger = set_logger('GAN')

# Hyper parameters
EPOCH = 1000
BATCH_SIZE = 64
LR_G = 0.0001
LR_D = 0.0001

# GPU
G_losses = []
D_losses = []
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
transform = transforms.ToTensor()
data_train = datasets.MNIST(root="../dataset",
                            transform=transform,
                            train=True,
                            download=True)
data_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss().to(device)
D_optim = torch.optim.Adam(D.parameters(), lr=LR_D)
G_optim = torch.optim.Adam(G.parameters(), lr=LR_G)

for epoch in range(EPOCH):
    print('start epoch {}...'.format(epoch+1))
    for i, (img, label) in enumerate(data_loader):
        
        z = torch.randn(img.shape[0], 100).to(device)
        
        real_img = img.view(-1, 784).to(device)
        fake_img = G(z).to(device)
        
        real_label = torch.ones(real_img.shape[0], 1).to(device)
        fake_label = torch.zeros(fake_img.shape[0], 1).to(device)
        
        prob_real = D(real_img)
        prob_fake = D(fake_img)

#         D_loss = - torch.mean(torch.log(prob_real) + torch.log(1. - prob_fake))
#         G_loss = torch.mean(torch.log(1. - prob_fake))

        D_input = torch.cat((prob_real, prob_fake), 0)
        D_target = torch.cat((real_label, fake_label), 0)
        
        D_loss = criterion(D_input, D_target)      #input(64), target(128)
        G_loss = criterion(prob_fake, real_label)

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
        save_image((reshape(fake_img).data.cpu()),
                    os.path.join('./result', 'sample_{}.png'.format(epoch+1)),
                    nrow=4,
                    padding=2)
        torch.save(G.state_dict(),
                    os.path.join('./checkpoint', 'G_{}.pth'.format(epoch+1)))
        torch.save(D.state_dict(),
                    os.path.join('./checkpoint', 'D_{}.pth'.format(epoch+1)))
plt_loss(G_losses, D_losses)
print('Done!!!')