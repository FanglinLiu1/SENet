import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F


class RML_CNN(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(2, 7), stride=1, padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(64 * 128, 256)
        self.dropout3 = nn.Dropout(p)
        self.fc2 = nn.Linear(256, classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
