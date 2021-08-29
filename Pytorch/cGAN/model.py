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
        
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, z, label):
        label = self.label_emb(label)
        y = torch.cat((z, label), 1)
        return self.model(y)

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
        
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, y, label):
        label = self.label_emb(label)
        x = torch.cat([y, label], 1)
        return self.model(x)