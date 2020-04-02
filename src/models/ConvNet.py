import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.Model import Model

def sigmoid(x):
    return 1. / (1 + torch.exp(-x))

class ConvNet(Model):
    """ A simple ConvNet for acoustic scene classification on spectrograms
        Model Architecture: Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification - https://arxiv.org/pdf/1608.04363.pdf
    """
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=24, kernel_size=5, stride=1
        )
        nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(kernel_size=(4,2), stride=(4,2))

        self.conv2 = nn.Conv2d(
            in_channels=24, out_channels=48, kernel_size=5, stride=1
        )
        nn.init.xavier_uniform_(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(kernel_size=(4,2), stride=(4,2))

        self.conv3 = nn.Conv2d(
            in_channels=48, out_channels=48, kernel_size=5, stride=1
        )
        nn.init.xavier_uniform_(self.conv3.weight)

        # fc1 assumes input is (batch_sz, 1, 128, 96)
        self.fc1 = nn.Linear(in_features=1632, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, batch):
        x = F.relu(self.conv1(batch))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        y = sigmoid(self.fc2(x))
        return y

    def forward_and_convert(self, x):
        x_torch = torch.DoubleTensor(x)
        m = nn.Sigmoid()
        y_torch = m(self.forward(x_torch))
        return y_torch.detach().numpy()
