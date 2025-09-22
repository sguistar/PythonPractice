# 导入相关模块
from __future__ import print_function

import os
import zipfile
from time import time

import cv2
import numpy as np
import torch
# from sklearn.utils.fixes import loguniform
from scipy.stats import loguniform
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch import nn

from ResNet50 import resnet50

"""
参数配置
"""
train_parameters = {
    "input_size": [3, 224, 224],  # 输入图片的shape
    "class_dim": 5,  # 分类数
    "src_path": "flower7595.zip",  # 原始数据集路径
    "target_path": "D:\\PycharmProjects",  # 要解压的路径
}

# 解压数据集，请学习zipfile的用法
extracting = zipfile.ZipFile(train_parameters["src_path"])  # 解压源数据
extracting.extractall(train_parameters["target_path"])  # 解压到指定路径

"""
获取图片数据及lable
返回值：all_data为图片数据； all_data_label为图片标签
"""


def get_data(target_path):
    """
    生成数据
    """

    # data_dir = 'work'
    data_dir = target_path
    # print(data_dir)
    all_data = []
    all_data_label = []
    target_names = []

    # 生成数据集标注文件
    # all_dirs = []

    count = -1  # 记录类别序号,因为有根目录，所以从-1开始，根目录序号为0
    for root, dirs, files in os.walk(
            data_dir
    ):  # 分别从文件地址中，读出根地址、子目录、文件名
        # print(root)
        # print(dirs)
        # print(files)
        # 如果当前为数据根目录，则dirs就是各个子文件名称组成的列表，将dirs赋值给target_names
        if root == data_dir:
            target_names = dirs

        # 遍历子文件夹，获取数据及标签
        for filename in files:
            if "jpg" in filename:  # 只保留jpg文件
                # all_data_list.append(os.path.join(root,filename)+'\t'+str(count-1)+'\n')
                img = cv2.imread(str(os.path.join(root, filename)))
                img = cv2.resize(img, (224, 224))  # 调整为同尺寸224*224
                img_gray = cv2.cvtColor(
                    img, cv2.COLOR_BGR2GRAY
                )  # 改为灰度图,颜色空间转换函数
                h, w = img_gray.shape
                img_col = img_gray.reshape(h * w)  # 改为一维数据
                # print(h,w)
                all_data.append(img_col)  # 存图片
                all_data_label.append(count)  # label
        count = count + 1
        # print(count)
    print("数据已生成！")
    return all_data, all_data_label, target_names, h, w


def get_data2(target_path):  # 另一种读取数据的方式，保留颜色信息
    data_dir = target_path
    all_data, all_labels, target_names, paths = [], [], [], []

    # 先确定类别顺序：根目录下的子目录 sorted
    subdirs = [
        d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))
    ]
    subdirs.sort()
    target_names = subdirs[:]  # 类名列表
    cls2idx = {name: i for i, name in enumerate(target_names)}
    exts = {".jpg", ".jpeg", ".png", ".bmp"}  # 兼容大小写
    exts = exts | {e.upper() for e in exts}

    for cls in subdirs:
        cls_dir = os.path.join(data_dir, cls)
        files = [
            f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f))
        ]
        files.sort()  # 文件名也排序，确保跨平台一致
        for filename in files:
            ext = os.path.splitext(filename)[1]
            if ext in exts:
                p = os.path.join(cls_dir, filename)
                img = cv2.imread(p)
                if img is None:
                    continue
                img = cv2.resize(img, (224, 224))
                # ✅ 建议保留颜色信息：直接展平 RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, c = img_rgb.shape
                img_col = img_rgb.reshape(h * w * c).astype(np.float32) / 255.0

                all_data.append(img_col)
                all_labels.append(cls2idx[cls])
                paths.append(p)

    print("数据已生成！样本数:", len(all_data))
    # 打印每类样本数，方便你两端对比
    uniq, cnts = np.unique(np.array(all_labels), return_counts=True)
    print({target_names[i]: int(c) for i, c in zip(uniq, cnts)})

    return all_data, all_labels, target_names, h, w


# 获取数据，标签，物体名称,图片高和宽
all_data, all_data_label, target_names, h, w = get_data2(
    os.path.join(train_parameters["target_path"], "flowers")
)

# 将列表改成数组
X = np.array(all_data)
X = X.astype(np.float32)
y = np.array(all_data_label)


# 画图函数
def plot_gallery(images, titles, h, w, n_row=4, n_col=4):
    """
    安全画图：自动识别灰度/RGB；若是[0,1]浮点则放大到[0,255]再uint8显示；
    如果误传了非图像向量（比如PCA特征），会优雅跳过并标注。
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n = min(n_row * n_col, len(images))
    plt.figure(figsize=(2.1 * n_col, 2.8 * n_row))
    plt.subplots_adjust(
        bottom=0, left=0.01, right=0.99, top=0.90, wspace=0.75, hspace=0.35
    )

    for i in range(n):
        ax = plt.subplot(n_row, n_col, i + 1)
        x = images[i]

        # ---- 自检：看看数值范围（只打印前1张，避免刷屏）
        if i == 0:
            print(
                f"[debug] img0 shape={x.shape}, dtype={x.dtype}, min={x.min():.4f}, max={x.max():.4f}"
            )

        if x.size == h * w:  # 灰度
            img = x.reshape(h, w)
            # 若是float且范围<=1，放大到[0,255]
            if img.dtype != np.uint8 and img.max() <= 1.0 + 1e-6:
                img_disp = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_disp = img.astype(np.uint8)
            ax.imshow(img_disp, cmap="gray", vmin=0, vmax=255)

        elif x.size == h * w * 3:  # RGB
            img = x.reshape(h, w, 3)
            if img.dtype != np.uint8 and img.max() <= 1.0 + 1e-6:
                img_disp = (img * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_disp = img.astype(np.uint8)
            ax.imshow(img_disp)  # RGB 不要 cmap

        else:
            ax.axis("off")
            ax.set_title("Non-image features")
            continue

        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])


# 划分数据
# 要求：1、3/4用于训练，1/4用于测试；2、random_state设为1，随机种子保证得到相同随机数，保证每次运行结果一致。
# ******************请大家补全代码******************************#
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# *************************************************************#

t0 = time()
"""
# ***********定义主成分数量，可尝试修改*******************
"""
n_components = 500  # 主成分个数

"""
#调用PCA降维，参数：
#1、n_components；
#2、svd_solver是指定奇异值分解SVD的方法，有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}
#3、whiten：bool类型，True或者False，默认为False。表示是否进行白化处理。
#要求：svd_solver设为'auto', whiten设为False。
#可尝试修改各参数
"""
# ******************请大家补全代码******************************#
pca = PCA(n_components, svd_solver="randomized", whiten=False, random_state=1)
pca.fit(X_train)

# *************************************************************#

# 将数据X_train转换成降维后的数据。
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
t1 = time()
print("done in %0.3fs" % (t1 - t0))
print(X_train_pca.shape)

""""""
print("Fitting the classifier to the training set")
t0 = time()

# 选取一系列可能是最优的超参数C和gamma，遍历找到最优值
param_grid = {
    "C": loguniform(1, 1e5),
    "gamma": loguniform(1e-4, 1e-1),
}
# ******************请大家补全代码******************************#
# 要求：
# 1、采用RandomizedSearchCV对上面定义的param_grid字典进行遍历；
# 2、分类器采用SVC；SVC中的kernel可尝试不同的核函数，{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’}, SVC中的class_weight设为"balanced"；
# 3、RandomizedSearchCV中的n_iter设为20，cv设为5；


clf = RandomizedSearchCV(
    SVC(kernel="poly", class_weight="balanced"),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=1,
)


# *************************************************************#

# clf = clf.fit(X_train_pca, y_train)
# clf = clf.fit(X_train, y_train)
# print("done in %0.3fs" % (time() - t0))
# print("Best estimator found by grid search:")
# print(clf.best_estimator_)

# 下面用训练好的分类器进行预测

# print("Predicting people's names on the test set")  # 在测试集中预测人们的名字
# t0 = time()
# y_pred = clf.predict(X_test_pca)  # 进行预测
# y_pred = clf.predict(X_test)  # 进行预测

# print("done in %0.3fs" % (time() - t0))

# print(classification_report(y_test, y_pred, target_names=target_names))  # 查准率/查全率/F1值/测试样本数


# print(classification_report(y_test, y_pred))  # 查准率/查全率/F1值/测试样本数
# print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

# plot the result of the prediction on a portion of the test set
# 在测试集的一部分上绘制预测结果


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]]
    true_name = target_names[y_test[i]]
    return " predicted: %s\n true:      %s" % (pred_name, true_name)


# 画图展示
# prediction_titles = [
#     title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])
# ]
# plot_gallery(X_test, prediction_titles, h, w)
#
# plt.show()

model = resnet50(num_classes=5)
print(model)  # 打印模型结构
# 训练模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("Using {} device training.".format(torch.cuda.get_device_name()))
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 60
for epoch in range(epochs):
    print("Epoch {}/{}".format(epoch + 1, epochs))
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for i in range(len(X_train)):
        inputs = torch.tensor(
            X_train[i], dtype=torch.float32).reshape(1, 3, 224, 224)
        labels = torch.tensor(y_train[i], dtype=torch.long).reshape(1)

        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(X_train)
    epoch_acc = running_corrects.double() / len(X_train)

    print("Loss: {:.4f} Acc: {:.4f}".format(epoch_loss, epoch_acc))
    
# 测试模型
model.eval()
y_pred = []
with torch.no_grad():
    for i in range(len(X_test)):
        inputs = torch.tensor(X_test[i],
                              dtype=torch.float32).reshape(1, 3, 224, 224)
        labels = torch.tensor(y_test[i], dtype=torch.long).reshape(1)

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred.append(preds.item())

print(classification_report(y_test, y_pred, target_names=target_names))
