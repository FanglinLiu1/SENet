import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RML_MCLDNet(nn.Module):
    def __init__(self, classes=11, p=0.5, tap=8):
        super(RML_MCLDNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.conv0 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn0 = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 5), stride=1, padding=(0, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p)
        self.dropout4 = nn.Dropout(p)
        self.dropout5 = nn.Dropout(p)

        self.lstm = nn.LSTM(64, 64, 2,
                            bias=True, batch_first=True, dropout=p, bidirectional=False)

        self.pool = nn.AdaptiveAvgPool2d((1, 64))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 1 * 64, 1024)
        self.dropout1 = nn.Dropout(p)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 11)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.relu(self.bn0(self.conv0(x)))

        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x)))
        x3 = F.relu(self.bn3(self.conv3(x)))

        x = x1 + x2 + x3
        x = x + F.relu(self.bn4(self.conv4(x)))

        x = x1 + x2 + x3 + self.dropout3(x)
        x = x + F.relu(self.bn5(self.conv5(x)))

        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

