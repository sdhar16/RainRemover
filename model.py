import torch
import torchvision
from torch import nn
from torch.autograd import Variable

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),  # b, 16, 10, 10
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),  # b, 8, 3, 3
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=1,padding = 1),  # b, 16, 5, 5
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),  # b, 8, 15, 15
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.ConvTranspose2d(32, 3, 3, stride=1, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

