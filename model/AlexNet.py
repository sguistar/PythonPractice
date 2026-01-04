from torch import nn
class AlexNet(nn.Module):
    def __init__(self, num_class=10):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.classifier = nn.Sequential(nn.Linear(in_features=384 * 5 * 5, out_features=1024),
                                               nn.ReLU(), nn.Dropout(0.5),
                                               nn.Linear(in_features=1024, out_features=256), nn.ReLU(),
                                               nn.Dropout(0.5))

    def forward(self, input):
        y = self.pool1(input)
        y = self.relu1(self.conv1(y))

        y = self.pool2(y)
        y = self.relu2(self.conv2(y))

        y = self.relu3(self.conv3(y))
        y = self.relu4(self.conv4(y))
        y = self.relu5(self.conv5(y))

        y = self.pool3(y)

        y = (nn.Flatten(y))
        y = self.classifier(y)
        return y