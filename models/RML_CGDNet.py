import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDropout(nn.Module):
    """Apply multiplicative 1-centered Gaussian noise.

    As it is a regularization layer, it is only active at training time.

    Args:
        rate (float): Drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        seed (int): Optional random seed to enable deterministic behavior.
    """

    def __init__(self, rate, seed=None):
        super(GaussianDropout, self).__init__()
        if not 0 <= rate <= 1:
            raise ValueError(
                f"Invalid value received for argument `rate`. Expected a float value between 0 and 1. Received: rate={rate}"
            )
        self.rate = rate
        self.seed = seed
        if rate > 0 and seed is not None:
            torch.manual_seed(seed)

    def forward(self, inputs, training=False):
        if training and self.rate > 0:
            stddev = math.sqrt(self.rate / (1.0 - self.rate))
            noise = torch.randn_like(inputs) * stddev + 1.0
            return inputs * noise
        return inputs

    def __repr__(self):
        return f"GaussianDropout(rate={self.rate}, seed={self.seed})"


class RML_CGDNet(nn.Module):
    def __init__(self, classes=11, p=0.8):
        super(RML_CGDNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout1 = nn.Dropout(p)
        self.gaussian_dropout1 = GaussianDropout(p)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout2 = nn.Dropout(p)
        self.gaussian_dropout2 = GaussianDropout(p)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4), padding=(0, 0))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout3 = nn.Dropout(p)
        self.gaussian_dropout3 = GaussianDropout(p)

        self.flatten = nn.Flatten()
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=2, bias=True, batch_first=True)
        self.dropout4 = nn.Dropout(p)
        self.gaussian_dropout4 = GaussianDropout(p)
        self.fc1 = nn.Linear(64 * 32, 1024)
        self.dropout5 = nn.Dropout(p)
        self.gaussian_dropout5 = GaussianDropout(p)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout6 = nn.Dropout(p)
        self.gaussian_dropout6 = GaussianDropout(p)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = self.pool1(x1)
        x1 = self.gaussian_dropout1(x1)

        x2 = F.relu(self.conv2(x1))
        x2 = self.pool2(x2)
        x2 = self.gaussian_dropout2(x2)

        x3 = F.relu(self.conv3(x2))
        x3 = self.pool3(x3)
        x3 = self.gaussian_dropout3(x3)

        x1 = self.conv4(x1)
        x1 = F.relu(x1)
        x1 = self.pool4(x1)
        x3 = self.gaussian_dropout4(x3)

        x = x1 + x3
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.dropout4(x)
        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
        x = self.fc3(x)
        return x
