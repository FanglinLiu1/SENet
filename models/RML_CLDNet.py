import torch
import torch.nn as nn
import torch.nn.functional as F


class RML_CLDNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_CLDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p)

        self.lstm = nn.LSTM(input_size=64, hidden_size=256, num_layers=1,
                            bias=True, batch_first=True, bidirectional=False)
        self.dropout4 = nn.Dropout(p)

        self.fc1 = nn.Linear(in_features=256, out_features=1024)
        self.dropout5 = nn.Dropout(p)
        self.fc2 = nn.Linear(in_features=1024, out_features=classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))

        x2 = F.relu(self.bn2(self.conv2(x1)))

        x3 = F.relu(self.bn3(self.conv3(x2)))

        x4 = F.relu(self.bn4(self.conv3(x3)))

        x = self.dropout4(x4) + x1
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x, _ = self.lstm(x)

        x, _ = torch.max(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x
