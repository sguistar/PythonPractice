import torch
from torch import nn
class MLP(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((32, 32))
        self.flatten = nn.Flatten()
        input_features = 3 * 32 * 32
        self.fc1 = nn.Linear(input_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.drop1 =  nn.Dropout(0.5)

        self.fc2 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5)

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.drop1(self.act1(self.bn1(self.fc1(x))))
        x = self.drop2(self.act2(self.bn2(self.fc2(x))))
        return self.head(x)