import torch
from torch import nn

device = torch.device('mps')
# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return Y.reshape(Y.shape[2:])


X = torch.rand(8,8)
#conv2d1 = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
conv2d2 = nn.Conv2d(1, 1, kernel_size=(3, 2), padding=(2, 1))
#print(comp_conv2d(conv2d1, X).shape) #(8-3+0+3)/3=2, (8-5+2x1+4)/4=2 => output size = (2,2)
print(comp_conv2d(conv2d2, X).shape) #(8+2x2-(3-1)-1)/1+1=10, (8+2x1-(2-1)-1)/1+1=9