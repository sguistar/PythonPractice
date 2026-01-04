import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.conv5 = DoubleConv(512, 1024)

        self.up6 = nn.Conv2d(1024, 512, kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(1024, 512)
        self.up7 = nn.Conv2d(512, 256, kernel_size=2, stride=2)
        self.conv7 = nn.Conv2d(512, 256)
        self.up8 = nn.Conv2d(256, 128, kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 128)
        self.up9 = nn.Conv2d(128, 64, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(128, 64)

        self.conv10 = nn.Conv2d(64, 5, kernel_size=1)

        def forward(self, x):
            # --------------向下卷积-------------------#
            c1 = self.conv1(x)
            p1 = self.pool1(c1)

            c2 = self.conv2(p1)
            p2 = self.pool2(c2)

            c3 = self.conv3(p2)
            p3 = self.pool3(c3)

            c4 = self.conv4(p3)
            p4 = self.pool4(c4)

            c5 = self.conv5(p4)

            # --------------向上卷积-------------------#
            up_6 = self.up6(c5)
            merge6 = torch.concat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
            c6 = self.conv6(merge6)

            up_7 = self.up7(c6)
            merge7 = torch.concat([up_7, c3], dim=1)
            c7 = self.conv7(merge7)

            up_8 = self.up8(c7)
            merge8 = torch.concat([up_8, c2], dim=1)
            c8 = self.conv8(merge8)

            up_9 = self.up9(c8)
            merge9 = torch.concat([up_9, c1], dim=1)
            c9 = self.conv9(merge9)

            # 输出卷积
            c10 = self.conv10(c9)

            return c10