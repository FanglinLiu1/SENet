import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class RML_DAE(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_DAE, self).__init__()
        self.conv = nn.Conv2d(1, 64, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.lstm = nn.LSTM(64, 64, 2,
                            bias=True, batch_first=True, dropout=p, bidirectional=False)

        self.flatten = nn.Flatten()

        self.dropout = nn.Dropout(p)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 256, 1024),
            nn.ReLU(),
            nn.LayerNorm(1024),
            nn.Dropout(p),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(p),
            nn.Linear(256, classes),
        )

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        x, _ = self.lstm(x)

        # x, _ = torch.max(x, dim=1)
        x = self.flatten(x)
        # x = F.relu(x)
        x = self.dropout(x)
        xc = self.classifier(x)
        return xc
