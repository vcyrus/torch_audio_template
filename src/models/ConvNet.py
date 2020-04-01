import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    return 1. / (1 + torch.exp(-x))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1(1, 24, 5)

    def forward(self, x):
        return y

    def forward_and_convert(self, x):
        x_torch = torch.DoubleTensor(x)
        m = nn.Sigmoid()
        y_torch = m(self.forward(x_torch))
        return y_torch.detach().numpy()
