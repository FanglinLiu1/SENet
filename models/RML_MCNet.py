import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class RML_MCNet(nn.Module):
    def __init__(self, classes=11, p=0.5):
        super(RML_MCNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 7), stride=1, padding=(0, 3))
        # self.bn1 = nn.BatchNorm2d(64)
        # self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        #
        # self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.bn2 = nn.BatchNorm2d(64)
        # self.pool2 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        #
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.bn3 = nn.BatchNorm2d(64)
        # self.pool3 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        # self.dropout3 = nn.Dropout(p)
        #
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.bn4 = nn.BatchNorm2d(64)
        # self.dropout4 = nn.Dropout(p)
        # self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        # self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        # self.bn5 = nn.BatchNorm2d(64)
        # self.dropout5 = nn.Dropout(p)
        # self.pool5 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        #
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(64 * 2 * 32, 1024)
        # self.dropout6 = nn.Dropout(p)
        # self.fc2 = nn.Linear(1024, 256)
        # self.dropout7 = nn.Dropout(p)
        # self.fc3 = nn.Linear(256, 11)
        self.conv = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=False)
        self.bn = nn.BatchNorm2d(64)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv2 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv3 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))
        self.dropout3 = nn.Dropout(p)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(64)
        self.dropout4 = nn.Dropout(p)
        self.pool4 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv5 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(64)
        self.dropout5 = nn.Dropout(p)
        self.pool5 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv6 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout6 = nn.Dropout(p)
        self.pool6 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv7 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn7 = nn.BatchNorm2d(64)
        self.dropout7 = nn.Dropout(p)
        self.pool7 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.conv8 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn8 = nn.BatchNorm2d(64)
        self.dropout8 = nn.Dropout(p)
        self.pool8 = nn.MaxPool2d((1, 2), stride=(1, 2), padding=(0, 0))

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 32, 1024)
        self.dropout6 = nn.Dropout(p)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 256)
        self.dropout7 = nn.Dropout(p)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 11)

    # def forward(self, x):
    #     x = F.relu(self.conv1(x))
    #     x = self.pool1(x)
    #
    #     x2 = F.relu(self.conv3(x))
    #     x2 = x + self.dropout3(x2)
    #
    #     x1 = F.relu(self.conv2(x))
    #     x1 = self.pool2(x1)
    #
    #     x3 = F.relu(self.conv4(x1))
    #     x3 = x1 + self.dropout4(x3)
    #
    #     x4 = F.relu(self.conv5(x2))
    #     x4 = self.pool5(x4)
    #
    #     x = x3 + self.dropout5(x4)
    #
    #     x = self.flatten(x)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.dropout6(x)
    #     x = self.fc2(x)
    #     x = F.relu(x)
    #     x = self.dropout7(x)
    #     x = self.fc3(x)
    #     return x

    def forward(self, x):
        # x = self.bn(self.conv(x))

        x = F.relu(self.bn1(self.conv1(x)))

        x1 = F.relu(self.bn2(self.conv2(x)))
        x1 = self.pool1(x1)

        x2 = F.relu(self.bn3(self.conv3(x)))
        x2 = self.pool2(x2)

        x = x1 + x2
        x = F.relu(self.bn4(self.conv4(x)))

        x = F.relu(self.bn5(self.conv5(x)))

        x1 = F.relu(self.bn6(self.conv6(x)))
        x1 = self.pool3(x1)

        x2 = F.relu(self.bn7(self.conv7(x)))
        x2 = self.pool4(x2)

        x = x1 + x2
        x = F.relu(self.bn8(self.conv8(x)))
        # x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout6(x)
        x = self.fc3(x)
        return x
