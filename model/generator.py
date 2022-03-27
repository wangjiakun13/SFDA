import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_chans, base_chans=None, out_features=None):
        super(Generator, self).__init__()
        self.fc = nn.Linear(in_chans, out_features)
        self.norm = nn.BatchNorm2d()
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.convT1 = nn.ConvTranspose2d(in_channels=out_features,
                                         out_channels=512,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.convT2 = nn.ConvTranspose2d(in_channels=512,
                                         out_channels=256,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.convT3 = nn.ConvTranspose2d(in_channels=256,
                                         out_channels=256,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.convT4 = nn.ConvTranspose2d(in_channels=256,
                                         out_channels=128,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1)
        self.conv = nn.Conv2d(in_channels=128,
                              out_channels=3,
                              kernel_size=3,
                              stride=1,
                              padding=1)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.convT1(x)
        x = self.norm(x)
        x = self.act1(x)
        x = self.convT2(x)
        x = self.norm(x)
        x = self.act1(x)
        x = self.convT3(x)
        x = self.norm(x)
        x = self.act1(x)
        x = self.convT4(x)
        x = self.norm(x)
        x = self.act1(x)
        x = self.conv(x)
        x = self.act2(x)

        return x
