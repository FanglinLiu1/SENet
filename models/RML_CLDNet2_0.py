import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RML_CLDNet2_0(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_CLDNet2_0, self).__init__()

        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn1 = nn.BatchNorm2d(256)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(2, 3), stride=1, padding=(0, 1), bias=True)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(p)
        self.conv3 = nn.Conv2d(256, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True)
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p)
        
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=1,
                            bias=True, batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(128, 256)
        self.dropout5 = nn.Dropout(p)
        self.fc2 = nn.Linear(256, classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        # x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        # x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        x = self.dropout3(x)
        x = F.relu(self.conv4(x))
        # x = self.bn4(x)
        x = self.dropout4(x)

        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x, _ = self.lstm(x)

        x, _ = torch.max(x, dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x
