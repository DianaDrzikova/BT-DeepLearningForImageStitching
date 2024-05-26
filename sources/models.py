import torch
import torch.nn as nn
import torch.nn.init as init

# Diana Maxima Drzikova

class OriginalHomography(nn.Module):
    def __init__(self):
        super(OriginalHomography, self).__init__()

        self.conv1 = nn.Conv2d(2,64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64,64, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(64,128, kernel_size=3, stride=2, padding=1)
        self.conv6 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d(128,128, kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(128*2*2, 256)
        self.fc2 = nn.Linear(256,9)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        self.out = nn.Sigmoid()

        self.pool = nn.MaxPool2d(2,2)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn1(self.conv4(x)))

        x = self.pool(x)

        x = self.relu(self.bn2(self.conv5(x)))
        x = self.relu(self.bn2(self.conv8(x)))

        x = self.pool(x)

        x = x.view(-1,128 * 2 * 2)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        x = x.resize(x.shape[0], 3,3)

        return self.out(x)
    


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
