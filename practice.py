import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import d2l

X = torch.randn(1,1,224,224)
net = nn.Sequential(nn.Conv2d(1,6,5), nn.BatchNorm2d(num_features=6),nn.ReLU(),
                    nn.Conv2d(6,16,5), nn.BatchNorm2d(num_features=16),nn.ReLU(),
                    nn.AdaptiveAvgPool2d((53,53)),
                    nn.Flatten(),
                    nn.Linear(16*53*53,120), nn.ReLU(),
                    nn.Linear(120,84), nn.ReLU(),
                    nn.Linear(84,10))

y = net(X)
print(y.shape)
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()

