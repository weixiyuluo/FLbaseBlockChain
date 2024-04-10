# -*- codeing = utf-8 -*-
# @Time : 2023/4/13 15:57
# @Author : 李国锋
# @File: client.py
# @Softerware:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms ##从torchvision中引入图像转换
from torchvision import datasets
import numpy as np


class Client():
    def __init__(self, model):
        self.batch_size = 64
        transform = transforms.Compose([transforms.ToTensor(),   #ToTensor将图像PIL中的字节转换成tensor，归一化; #Normalize将数据进行标准化，1是均值，2是标准差 #3channel
                                        transforms.Normalize(mean=[0.5], std=[0.5])])
        #data set
        self.data_train = datasets.MNIST(root="./data/",
                                    transform=transform,
                                    train=True,
                                    download=True)

        self.data_test = datasets.MNIST(root="./data/",
                                   transform=transform,
                                   train=False)
        #data loader
        self.data_loader_train = torch.utils.data.DataLoader(dataset=self.data_train,
                                                        batch_size=64,
                                                        shuffle=True,
                                                        num_workers=0)

        self.data_loader_test = torch.utils.data.DataLoader(dataset=self.data_test,
                                                       batch_size=64,
                                                       shuffle=False,
                                                       num_workers=0)

        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.75)
        self.cost = torch.nn.CrossEntropyLoss().cuda()  #交叉熵
        self.gradForAggre = []            #
        self.loss = 0.0



    def getGrad(self):
        layers_global = np.zeros(7850)   #存储一次完整训练的梯度
        running_loss = 0.0
        for data in self.data_loader_train:  # (图片28*28*1*64，图片标签64*1) #打乱的训练数据集
            X_train, y_train = data
            X_train = X_train.cuda()
            y_train =y_train.cuda()
            outputs = self.model(X_train)
           # _, pred = torch.max(outputs.data, 1)
            self.optimizer.zero_grad()          #梯度清零，防止memory爆炸
            loss = self.cost(outputs, y_train)  #softmax+计算loss #y_train是标签为一个长度为batchsize的Tensor在[0, num_class-1]num_class为类别数
            loss.backward()                     #计算梯度
            running_loss += loss.item()
            #running_correct += torch.sum(pred == y_train.data)
            layers = np.zeros(0)          #存储一个batchsize完成训练后的梯度
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # print(param)
                    layers = np.concatenate((layers, param.grad.cpu().flatten()), axis=None)  #把梯度（tensor）变成一个一维数组#gup->cpu
            layers_global = layers + layers_global  #梯度累加
            #print("training accuracy is: ", 100 * running_correct / len(self.data_train))
            #print(layers_global)
        return layers_global

    def updateModel(self, modelWeights):  # 将收到的全局模型替换本地模型

        layers = self.model.reshape(modelWeights)  # 把数组变为张量  变成w和b  #变成gpu里面的张量
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:         # 需要计算梯度的参数
                #param =param.cpu()
                param.data = layers[layer]  #len(layers)=2#更新一个权重一个偏置
                #param = param.cuda()
                layer += 1

    # Called when an aggregator receives a new gradient
    def reciveGardToAggre(self, gradient):
        # Reshape into original tensor
        layers = self.model.reshape(gradient)
        self.gradForAggre.append(layers)
       # print(self.gradForAggre)


    # 当梯度聚合好后在区块链中调用
    def simpleStep(self, gradient):  #沿梯度方向更新
        print("Simple step")
        layers = self.model.reshape(gradient)    #将梯度数组转换为张量  #收到的梯度的聚合
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = layers[layer]   #梯度
                layer += 1
        # 减去梯度，更新模型
        self.optimizer.step()

    # 当聚合者收到足够的梯度时利用收到的梯度更新模型  #防止掉线
    # def step(self):
    #     print("Simple step 1")
    #     # 聚合梯度
    #     for i in range(1, len(self.gradForAggre)):   #i从1开始，梯度累加到0的位置
    #         gradients = self.gradForAggre[i]   #存的是张量，里面是w和b的梯度
    #         for g, gradient in enumerate(gradients):  #每个梯度一一个二维数组型式存储，分别是w和b的梯度
    #             self.gradForAggre[0][g] += gradient
    #
    #     # 梯度平均
    #     for g, gradient in enumerate(self.aggregatedGradients[0]):
    #         gradient /= len(self.aggregatedGradients)
    #
    #     #simpleStep()
    #     layer = 0
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             param.grad = self.aggregatedGradients[0][layer]
    #             layer += 1
    #
    #     self.optimizer.step()
    #     self.aggregatedGradients = []

    def getTrainAccur(self):
        running_correct = 0
        for data in self.data_loader_train:  # (图片28*28*1*64，图片标签64*1) #打乱的训练数据集
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = self.model(X_train)
            _, pred = torch.max(outputs.data, 1)
            running_correct += torch.sum(pred == y_train.data)
        return (100 * running_correct / len( self.data_train)).__float__()

    def getTestAccur(self):
        running_correct = 0
        for data in self.data_loader_test:
            X_train, y_train = data
            X_train =  X_train.cuda()
            y_train = y_train.cuda()
            outputs = self.model(X_train)
            _, pred = torch.max(outputs.data, 1)
            running_correct += torch.sum(pred == y_train.data)
        t =(100 * running_correct / len( self.data_test)).__float__()
        return t

    def getModelWeights(self):  #获取权重和偏置的值并转化为数组   #……
        layers = np.zeros(0)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers = np.concatenate((layers, param.data.cpu().flatten()), axis=None)
        return layers

    def getLoss(self):
        return self.loss

    def getModel(self):
        return self.model































