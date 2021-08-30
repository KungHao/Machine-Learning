import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(100+10, 256),    # 100+10
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784)
        )
        
        self.layer = nn.Sequential(
            nn.Linear(100+10, 1)
        )

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, 1, 1)
        )
        
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, z, label):
        label = self.label_emb(label)
        y = torch.cat((z, label), 1)
        input = self.layer(y)
        input = input.view(-1, 1, 1, 1)
        return self.model(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784+10, 512),     #784+10
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.layer = nn.Sequential(
            nn.Linear(784+10, 1)
        )
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Linear(128, 4096),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, y, label):
        label = self.label_emb(label)
        x = torch.cat([y, label], 1)
        input = self.layer(x)
        input = input.view(-1, 1, 32, 32)
        return self.model(input)