import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


def cal1(x):
    return torch.cos(x)

def cal2(x):
    return torch.sin(x)


class RML_PET_CGDNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_PET_CGDNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 128, 1)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), padding=(0, 1))
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), padding=(0, 3))
        self.gru = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(1, classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x1 = self.flatten(x)
        x1 = self.fc1(x1)

        cosx = cal1(x1)
        sinx = cal2(x1)

        x11 = x * cosx
        x12 = x * sinx
        x21 = x * cosx
        x22 = x * sinx

        y1 = x11 + x12
        y2 = x21 - x22

        y = y1 + y2

        x = self.fc2(y)
        return x
