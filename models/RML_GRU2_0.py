import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RML_GRU2_0(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_GRU2_0, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.gru = nn.GRU(input_size=64, hidden_size=256, num_layers=2, bias=True, batch_first=True, dropout=p)
        self.fc = nn.Linear(256, classes)

    def forward(self, x):
        x = self.bn(self.conv(x))

        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x, _ = self.gru(x)
        x, _ = torch.max(x, dim=1)

        x = self.fc(x)
        return x
