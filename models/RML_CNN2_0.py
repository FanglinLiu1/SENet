import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class RML_CNN2_0(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_CNN2_0, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(2, 7), stride=1, padding=(0, 3), bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout1 = nn.Dropout(p)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout2 = nn.Dropout(p)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout3 = nn.Dropout(p)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(512)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout4 = nn.Dropout(p)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512 * 8, 1024)
        self.dropout5 = nn.Dropout(p)
        self.fc2 = nn.Linear(1024, classes)

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
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x
