import torch
import torch.nn as nn
import torchvision
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.init as init


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 4, 3, stride=2, padding=1)

        # Decoder layers
        self.conv4 = nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.LeakyReLU()
        
        self.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # Encoder
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        encoder = nn.functional.relu(self.conv3(x))

        x = self.dropout(x)

        # Decoder
        x = nn.functional.relu(self.conv4(encoder))
        x = nn.functional.relu(self.conv5(x))
        decoder = torch.sigmoid(self.conv6(x))

        return decoder, encoder