import torch
from torchvision import datasets, transforms
from numpy import compat
import matplotlib.pyplot as plt
import os
import torchvision
import numpy as np
from torch.autograd import Variable
import time

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])
batch_size = 10
# Data set
data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

# Data loader
data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=64,
                                                shuffle=True,
                                                num_workers=0)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=64,
                                               shuffle=True,
                                               num_workers=0)


# class Model(torch.nn.Module):
#
#     def __init__(self):
#         super(Model, self).__init__()
#         self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#                                          torch.nn.ReLU(),
#                                          torch.nn.MaxPool2d(stride=2, kernel_size=2))
#         self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
#                                          torch.nn.ReLU(),
#                                          torch.nn.Dropout(p=0.5),
#                                          torch.nn.Linear(1024, 10))
#
#     def forward(self, x):
#         x = self.conv1(x)
#         # x = self.conv2(x)
#         x = x.view(-1, 14 * 14 * 128)
#         x = self.dense(x)
#         return x
#
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(28 * 28, 10) # 7840 + 10
        #self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, 784) #28*28*1*64/784, 784
        x = self.linear(x) #28*28*1*64/784, 10
        #x = self.softmax(x)
        return x


model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001,momentum=0.2)
# print(loss2)
# sig = torch.nn.Sigmoid()
cost = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 1
time_start = time.time()
layers_global = np.zeros(7850)
# model.load_state_dict(torch.load('model_parameter.pkl'))
# model.load_state_dict(torch.load('model_parameter_watermark.pkl'))
for epoch in range(n_epochs):
    running_loss = 0.0
    running_correct = 0
    print("Epoch {}/{}".format(epoch+1, n_epochs))
    print("-" * 10)
    for data in data_loader_train:#(图片28*28*1*64，图片标签64*1)20000 INT 20000/64
        X_train, y_train = data
        X_train, y_train = Variable(X_train), Variable(y_train)
        outputs = model(X_train)
        _, pred = torch.max(outputs.data, 1)
        optimizer.zero_grad()
        loss = cost(outputs, y_train)
        loss.backward()
        optimizer.step()
        running_loss += loss.data
        running_correct += torch.sum(pred == y_train.data)
        layers = np.zeros(0)
        for name, param in model.named_parameters():
            if param.requires_grad:
                # print(param)
                layers = np.concatenate((layers, param.grad.flatten()), axis=None)
        layers_global = layers+layers_global
    print("training accuracy is: ", (100 * running_correct / len(
           data_train)).__float__())
print(layers_global)
time_end = time.time()
print("training time is: ", time_end - time_start)

# model.eval()
# for data in data_loader_test:
#     X_test, Y_test = data
#     X_test, y_test = Variable(X_test), Variable(Y_test)
#     outputs = model(X_test)
#     _, pred = torch.max(outputs.data, 1)
#     testing_correct += torch.sum(pred == y_test.data)
# print("Loss is:{:.4f}, Water_loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(
#         running_loss / len(data_train),
#         water_loss / len(data_train),
#         100 * running_correct / len(
#             data_train),
#         100 * testing_correct / len(
#             data_test)))
