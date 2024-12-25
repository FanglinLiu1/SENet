import torch
import torch.nn as nn
import torch.nn.functional as F


class RML_LSTM(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=256, layer_dim=2, output_dim=11, p=0.5):
        super(RML_LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.lstm = nn.LSTM(64, 64, 2,
                            bias=True, batch_first=True, dropout=p, bidirectional=False)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 256, 1024)
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(256, 11)

    def forward(self, x):
        x = self.bn(self.conv(x))

        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
