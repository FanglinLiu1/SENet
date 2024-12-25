import torch.nn as nn
import torch.nn.functional as F


class RML_DenseNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_DenseNet, self).__init__()

        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p)

        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 1024)
        self.dropout5 = nn.Dropout(p)
        self.fc2 = nn.Linear(1024, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout6 = nn.Dropout(p)
        self.fc3 = nn.Linear(256, 11)

    def forward(self, x):
        x = self.bn(self.conv(x))

        x1 = F.relu(self.bn1(self.conv1(x)))
        # x1 = self.dropout1(x1)

        x2 = F.relu(self.bn2(self.conv2(x + self.dropout1(x1))))
        # x2 = self.dropout2(x2)

        x3 = F.relu(self.bn3(self.conv3(x + x1 + self.dropout2(x2))))
        # x3 = self.dropout3(x3)

        x = F.relu(self.bn4(self.conv4(x + x1 + x2 + self.dropout3(x3))))

        x = self.pool(x)
        x = self.flatten(x)

        resuial = x
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))

        x = self.ln1(resuial + self.dropout6(x))
        x = self.fc3(x)
        return x
