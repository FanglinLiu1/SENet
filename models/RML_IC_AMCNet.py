import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class GaussianDropout(nn.Module):
    def __init__(self, p: float):
        """
        初始化高斯Dropout模块。

        参数:
        - p (float): 高斯Dropout的标准差比例（相当于Dropout概率）。
        """
        super(GaussianDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.p + 1.0
            return x * noise
        else:
            return x


class RML_IC_AMCNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_IC_AMCNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn5 = nn.BatchNorm2d(64)

        self.pool1 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

        self.gaussian_noise = GaussianDropout(p)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 32, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        # x = self.bn(self.conv(x))
        x = F.relu(self.bn1(self.conv1(x)))

        x = x + F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x + F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        x = F.relu(self.bn4(self.conv4(x)))

        # x = x + F.relu(self.bn2(self.conv2(x)))
        # x = self.pool1(x)
        #
        # x = F.relu(self.bn3(self.conv3(x)))
        #
        # x = x + F.relu(self.bn4(self.conv4(x)))
        # x = self.pool2(x)

        x = self.flatten(x)

        x = F.relu(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
