from torch import nn

class ConvPool(nn.Module):
    '''卷积+池化'''

    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 pool_size,
                 pool_stride,
                 groups,
                 conv_stride=1,
                 conv_padding=1,

                 ):
        super(ConvPool, self).__init__()

        for i in range(groups):
            self.add_sublayer(  # 添加子层实例
                'bb_%d' % i,
                nn.Conv2D(  # layer
                    in_channels=num_channels,  # 通道数
                    out_channels=num_filters,  # 卷积核个数
                    kernel_size=filter_size,  # 卷积核大小
                    stride=conv_stride,  # 步长
                    padding=conv_padding,  # padding
                )
            )
            self.add_sublayer(
                'relu%d' % i,
                nn.ReLU()
            )
            num_channels = num_filters

        self.add_sublayer(
            'Maxpool',
            nn.MaxPool2D(
                kernel_size=pool_size,  # 池化核大小
                stride=pool_stride  # 池化步长
            )
        )

    def forward(self, inputs):
        x = inputs
        for prefix, sub_layer in self.named_children():
            # print(prefix,sub_layer)
            x = sub_layer(x)
        return x

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()

        self.convpool1 = ConvPool(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            pool_size=2,
            pool_stride=2,
            groups=2,
            conv_stride=1,
            conv_padding=1)

        self.convpool2 = ConvPool(
            num_channels=64,
            num_filters=128,
            filter_size=3,
            pool_size=2,
            pool_stride=2,
            groups=2,
            conv_stride=1,
            conv_padding=1
        )

        self.convpool3 = ConvPool(
            num_channels=128,
            num_filters=256,
            filter_size=3,
            pool_size=2,
            pool_stride=2,
            groups=3,
            conv_stride=1,
            conv_padding=1)

        self.convpool4 = ConvPool(
            num_channels=256,
            num_filters=512,
            filter_size=3,
            pool_size=2,
            pool_stride=2,
            groups=3,
            conv_stride=1,
            conv_padding=1)

        self.convpool5 = ConvPool(
            num_channels=512,
            num_filters=512,
            filter_size=3,
            pool_size=2,
            pool_stride=2,
            groups=3,
            conv_stride=1,
            conv_padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000),
        )

    def forward(self, inputs, label=None):
        y = self.convpool1(inputs)
        y = self.convpool2(y)
        y = self.convpool3(y)
        y = self.convpool4(y)
        y = self.convpool5(y)
        y = nn.Flatten(y)
        y = self.classifier(y)

        return y
