# -*- codeing = utf-8 -*-
# @Time : 2023/4/13 15:46
# @Author : 李国锋
# @File: softmax_model.py
# @Softerware:

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class SoftmaxModel(nn.Module):   #一层全连接神经网络
    def __init__(self):  # 构造函数，初始化默认调用
        super(SoftmaxModel, self).__init__()  # 自己类的名称，self
        self.linear = nn.Linear(784, 10).cuda()  # 784维输入，10维输出 #linear包含权重和偏置 #继承自model，能自动反向传播
        self.D_in = 784
        self.D_out = 10

    def forward(self, x):  # 前馈过程调用
        x = x.view(-1, 784)

        #x = np.reshape(x,(x.shape[0], self.D_in) ) #x.view(-1,784) #(x.shape[0], self.D_in) x的第一维度的长度,相当于行数 #把x转维x.shape[0]行，784列的矩阵

        return self.linear(x)  # __call__(),可调用的对象，call中调用了forward()


    def reshape(self, flat_gradient):  # 把数组转换成张量
        layers = []  #张量
        layers.append(
            torch.from_numpy(np.reshape(flat_gradient[0:784 * 10], (10, 784))).type(
                torch.FloatTensor).cuda())  # 权重数组 #[784，10]  #y = (xw)^t+b
        layers.append(torch.from_numpy(flat_gradient[784 * 10:784 * 10 + 10]).type(
            torch.FloatTensor).cuda())  #偏置数组
        return layers

