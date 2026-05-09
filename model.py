import torch
import torch.nn as nn
import torch.nn.functional as F

class PneumoniaCNN(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        self.conv1 = nn.Conv2d(3, 23, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(23)

        self.conv2 = nn.Conv2d(23, 46, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(46)

        self.conv3 = nn.Conv2d(46, 92, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(92)

        self.conv4 = nn.Conv2d(92, 184, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(184)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(184, 64)

        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.gap(x)

        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.fc2(x)

        return x