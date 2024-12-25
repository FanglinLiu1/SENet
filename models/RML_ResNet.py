import torch.nn as nn
import torch.nn.functional as F


class RML_ResNet(nn.Module):
    def __init__(self, num_classes=11, p=0.5):
        super(RML_ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)

        self.pool = nn.AdaptiveAvgPool2d((2, 32))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 32, 1024)
        self.dropout3 = nn.Dropout(p)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout4 = nn.Dropout(p)
        self.fc3 = nn.Linear(256, 11)

    def forward(self, x):
        x = self.bn(self.conv(x))

        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = residual + self.dropout1(x)

        residual = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = residual + self.dropout2(x)

        x = self.pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc3(x)
        return x
