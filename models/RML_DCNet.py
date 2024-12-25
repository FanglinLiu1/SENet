import torch
import torch.nn as nn
import torch.nn.functional as F


class RML_DCNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_DCNet, self).__init__()

        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, bias=False, padding=(0, 0))
        self.bn = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(p)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(p)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn3 = nn.BatchNorm2d(64)
        self.dropout3 = nn.Dropout(p)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p)

        self.conv5 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout5 = nn.Dropout(p)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout6 = nn.Dropout(p)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(p)
        self.conv8 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn8 = nn.BatchNorm2d(64)
        self.dropout8 = nn.Dropout(p)

        self.conv9 = nn.Conv2d(64, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn9 = nn.BatchNorm2d(64)
        self.dropout9 = nn.Dropout(p)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn10 = nn.BatchNorm2d(64)
        self.dropout10 = nn.Dropout(p)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=(0, 0))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 32, 1024)
        self.dropout13 = nn.Dropout(p)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout14 = nn.Dropout(p)
        self.fc3 = nn.Linear(256, classes)

    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.bn4(self.conv4(x1)))

        x2 = F.relu(self.bn5(self.conv5(x)))
        x2 = F.relu(self.bn6(self.conv6(x2)))
        x2 = F.relu(self.bn7(self.conv7(x2)))
        x2 = F.relu(self.bn8(self.conv8(x2)))

        x = self.dropout8(x1 + x2) + self.bn(self.conv(x))

        x = F.relu(self.conv9(x))
        x = self.dropout9(x)
        x = self.pool1(x)
        x = F.relu(self.conv10(x))
        x = self.dropout10(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout13(x)
        x = F.relu(self.fc2(x))
        x = self.dropout14(x)
        x = self.fc3(x)
        return x
