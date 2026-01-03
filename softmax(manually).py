import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
from IPython import display
from utils import d2l

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ========= 1. 画图全局配置 =========
backend_inline.set_matplotlib_formats('svg')  # 输出 SVG，放大不糊
plt.rcParams['figure.figsize'] = (3.5, 2.5)  # 图像默认尺寸


# ========= 2. 工具类 =========
class Accumulator:
    """在 n 个变量上累加求和，用于统计 loss、准确率等"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        # 支持一次传入多个标量：loss_sum, correct_sum, sample_sum
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Animator:
    """
    训练动态可视化：每次 add() 会清空坐标轴并重绘
    legend: 曲线标签；fmts: 线型/颜色; X,Y: 二维列表按曲线分组
    """

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # 记录配置信息
        self.xlabel, self.ylabel = xlabel, ylabel
        self.legend = legend
        self.xlim, self.ylim = xlim, ylim
        self.xscale, self.yscale = xscale, yscale
        self.fmts = fmts
        self.X, self.Y = [], []  # 保存每条曲线的坐标

    def config_axes(self):
        """统一设置坐标系外观"""
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xscale(self.xscale)
        self.ax.set_yscale(self.yscale)
        if self.xlim: self.ax.set_xlim(self.xlim)
        if self.ylim: self.ax.set_ylim(self.ylim)
        if self.legend: self.ax.legend(self.legend)
        self.ax.grid()

    def add(self, x, y):
        """
        将新的 (x, y) 数据点追加到曲线中并实时刷新
        x, y 可以是标量或列表；若 y 是多条曲线，则 x 将被广播
        """
        if not hasattr(y, '__len__'):
            y = [y]
        if not hasattr(x, '__len__'):
            x = [x] * len(y)
        if not self.X:
            self.X = [[] for _ in y]
            self.Y = [[] for _ in y]
        for i, (xi, yi) in enumerate(zip(x, y)):
            self.X[i].append(xi)
            self.Y[i].append(yi)

        # 清空并重画
        self.ax.cla()
        for x_vals, y_vals, fmt in zip(self.X, self.Y, self.fmts):
            self.ax.plot(x_vals, y_vals, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)  # 实时刷新但不堆叠图像


# ========= 3. 模型与损失 =========
def softmax(X):
    """数值稳定版 softmax，按行做归一化"""
    X_exp = torch.exp(X - X.max(dim=1, keepdim=True)[0])  # 减去 max 防溢出
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def net(X):
    """前向计算：X shape (batch, 1, 28, 28)"""
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


def cross_entropy(y_hat, y):
    """交叉熵损失：取对应类别概率后加 log"""
    return -torch.log(y_hat[range(len(y_hat)), y] + 1e-12)  # 平滑 ε 防 log(0)


def accuracy(y_hat, y):
    """计算批量准确率（标量）"""
    if y_hat.ndim > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.sum())


# ========= 4. 评估 & 训练函数 =========
def evaluate_accuracy(net, data_iter):
    """整集推理准确率（no_grad 下加速）"""
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(50)  # correct, total
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    """单 epoch 训练，返回 avg_loss, avg_acc"""
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(50)  # loss_sum, correct, total
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        # 反向传播
        updater.zero_grad()
        l.mean().backward()
        updater.step()
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', ylabel='metrics',
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))


# ========= 5. 推理 + 可视化 =========
def predict(net, test_iter, n=6):
    X, y = next(iter(test_iter))
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [t + '\n' + p for t, p in zip(trues, preds)]
    d2l.show_images(X[:n].reshape((n, 28, 28)), 1, n, titles=titles[:n])


# ========= 6. 数据集 =========
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(root="../data",
                                                train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data",
                                               train=False, transform=trans, download=True)
batch_size = 256
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False)

# ========= 7. 参数初始化 =========
num_inputs, num_outputs = 28 * 28, 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

# ========= 8. 训练入口 =========
updater = torch.optim.SGD([W, b], lr=0.1)  # 使用 SGD 更新裸张量
num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
predict(net, test_iter)  # 随机展示若干预测结果
