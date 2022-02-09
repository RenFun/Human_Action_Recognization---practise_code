# Author:RenFun
# File:Load_data.py
# Time:2022/01/22


# 加载本地图片数据集
import os
import cv2
import torch
import torchvision
import csv
import numpy as np
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")


# 数据转换函数
def get_trans():
    transform = torchvision.transforms.Compose(
        [
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5])
        ]
    )
    return transform


# 定义一个类，有关于数据集
class DataSet(data.Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k)for k in imgs]
        # self.transforms = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        # 打开图片
        pil_img = Image.open(img_path)
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            pil_img = np.asarray(pil_img)
            data = torch.from_numpy(pil_img)
        return data

    def __len__(self):
        return len(self.imgs)


# 实例化后得到的数据是tensor类型，规模是1*120*160，即 channels*height*width
# dataset = DataSet('D:\DataSet\KTH\Data&Label.csv')


# 加载数据：通过读取.csv文件从而获得样本的地址
dataset = read_csv('D:\DataSet\KTH\Data&Label.csv', encoding='utf-8', header=None)
# 改变类型，289715*2的数组
dataset = np.array(dataset)
# 第一列为样本地址
DATA_PATH = dataset[:, 0]
# 第二列为样本标签
LABEL = dataset[:, 1]
# 使用opencv的中cv2.imread读入，参数为图片路径和读取方式（默认为彩色图片，以BGR格式），由于样本本身就是灰度图，所以此处使用彩色模式读取图片得到的就是灰度图
data = cv2.imread(DATA_PATH[0], cv2.IMREAD_COLOR)
data = cv2.resize(data, (80, 60), interpolation=cv2.INTER_AREA)
# plt.imshow(data)
# plt.show()
# 获取标签
label = LABEL[0]
# print(label)
# 使用PIL.Image.open读入，由于样本本身是灰度图，所以以BGR格式读入后色彩会发生改变
img = Image.open(DATA_PATH[0])
img = np.array(img)
# print(img.shape)
# plt.imshow(img)
# plt.show()
# 读取.csv文件的每一行数据，获得样本及其标签
kth_data = []
kth_label = []
with open('D:\DataSet\KTH\Data&Label.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        temp1 = cv2.imread(row[0], cv2.IMREAD_COLOR)
        # 改变图像规模，以便于输入到网络模型中
        temp1 = cv2.resize(temp1, (28, 28), interpolation=cv2.INTER_AREA)
        # temp1 = Image.open(row[0])
        # temp1 = np.array(temp1)
        temp2 = row[1]
        kth_data.append(temp1)
        kth_label.append(temp2)
kth_data = np.array(kth_data)
kth_label = np.array(kth_label)
# 对标签进行编码，0~n-1
kth_label = preprocessing.LabelEncoder().fit_transform(kth_label)
# 改变图片格式以便于将数据输入到网络模型中：由n*h*w*c 改为n*c*h*w
kth_data = kth_data.transpose([0, 3, 1, 2])
# 数组切片划出训练集和测试集
kth_train_data = kth_data[:750, :, :, :]
kth_train_label = kth_label[:750]
kth_test_data = kth_data[750:, :, :, :]
kth_test_label = kth_label[750:]
zipped = zip(kth_data, kth_label)
count = 0
for data, label in zip(kth_data, kth_label):
    count += 1
    # print(data, label)
print(count)
zipped = list(zipped)
zipped = np.array(zipped)
print(zipped.shape)
# stack = np.hstack((kth_data, kth_label))
# concatenate = np.concatenate((kth_data, kth_label), axis=1)
# zipped = zipped.astype(float)
# DATA = pd.DataFrame(zipped, dtype=np.float64)
# DATA = torch.from_numpy(zipped)
# 不同维数的数组不能进行堆叠
# 将数组类型改变为tensor类型
kth_data = torch.FloatTensor(kth_data)
kth_label = torch.IntTensor(kth_label)
# 转变类型
kth_train_data = torch.FloatTensor(kth_train_data)
kth_train_label = torch.IntTensor(kth_train_label)
kth_test_data = torch.FloatTensor(kth_test_data)
kth_test_label = torch.IntTensor(kth_test_label)
# print(type(kth_data))
