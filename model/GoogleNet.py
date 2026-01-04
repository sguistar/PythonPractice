import torch
from torch import nn
import torch.nn.functional as F
#构建模型
class Inception(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4):
        super(Inception, self).__init__()
        #路线1，卷积核1x1
        self.route1x1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        #路线2，卷积层1x1、卷积层3x3
        self.route1x1_2 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.route3x3_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        #路线3，卷积层1x1、卷积层5x5
        self.route1x1_3 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.rout5x5_3 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        #路线4，池化层3x3、e卷积层1x1
        self.route3x3_4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.route1x1_4 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        route1 = F.relu(self.route1x1_1(x))
        route2 = F.relu(self.route3x3_2(F.relu(self.route1x1_2(x))))
        route3 = F.relu(self.route5x5_3(F.relu(self.route1x1_3(x))))
        route4 = F.relu(self.route1x1_4(self.route3x3_4(x)))
        out = [route1, route2, route3, route4]
        return torch.concat(out, dim=1)  #在通道维度(axis=1)上进行连接

def BasicConv2d(in_channels, out_channels, kernel, stride=1, padding=0):
    layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                nn.BatchNorm2d(out_channels, eps=1e-3),
                nn.ReLU())
    return layer


class GoogLeNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(
            BasicConv2d(in_channels=3, out_channels=64, kernel=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = nn.Sequential(
            BasicConv2d(in_channels=64, out_channels=64, kernel=1),
            BasicConv2d(in_channels=64, out_channels=192, kernel=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = nn.Sequential(
            Inception(in_channels=192, c1=64, c2=(96, 128), c3=(16, 32), c4=32),
            Inception(in_channels=256, c1=128, c2=(128, 192), c3=(32, 96), c4=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b4 = nn.Sequential(
            Inception(480, 192, (96, 208), (16, 48), 64),
            Inception(512, 160, (112, 224), (24, 64), 64),
            Inception(512, 128, (128, 256), (24, 64), 64),
            Inception(512, 112, (144, 288), (32, 64), 64),
            Inception(528, 256, (160, 320), (32, 128), 128),
            nn.MaxPool2d(3, 2))

        self.b5 = nn.Sequential(
            Inception(832, 256, (160, 320), (32, 128), 128),
            Inception(832, 384, (182, 384), (48, 128), 128),
            nn.AdaptiveAvgPool2d((1, 1)))

        self.flatten = nn.Flatten()
        self.b6 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.flatten(x)
        x = self.b6(x)
        return x