# -*- codeing = utf-8 -*-
# @Time : 2023/4/13 17:02

# @File: client_obj.py
# @Softerware:

from __future__ import division



import numpy as np
import client
import pdb
from softmax_model import SoftmaxModel
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from scipy.interpolate import lagrange







expected_iters = 100

epsilon = 2  #差分隐私预算
delta = 0.00001

batch_size = 64






np.set_printoptions(threshold=7851)  #防止传递省略号
writer = SummaryWriter("new01")

def init():

    global  myclient
    model = SoftmaxModel().cuda()
    myclient = client.Client(model)

    global samples  #存储噪声
    samples = []

    sigma = np.sqrt(2 * (np.log(1.25 / delta))) / epsilon
    noise = sigma * np.random.randn(batch_size, expected_iters, 7850)  # 10，100，7850
    samples = np.sum(noise, axis=0)  # 100，7850


def getNoise(itera):  #初始化时任务发布者将此噪声上传到创世区块

    return (-1 / batch_size) * samples[ itera% expected_iters]  #-1为了与梯度方向相匹配

def privateFun(ww):   #下载全局模型并计算梯度
    global myclient
    weights = np.array(ww)
    myclient.updateModel(weights)  #将模型更新为全局模型
    return (-1) * myclient.getGrad()  #返回计算的梯度

def updateModel(modelWeights):   #只将全局模型替换本地模型
    global myclient
    myclient.updateModel(modelWeights)

def simpleStep(gradient):  #进行一次梯度更新#用传入的梯度数组
    global myclient
    myclient.simpleStep(gradient)

def getTestAccur(ww,i):    #获得模型的精确度
    global myclient
    weights = np.array(ww)
    myclient.updateModel(weights)
    t = myclient.getTestAccur()
    writer.add_scalar("our scheme test accuracy",t,i)
    return t

def ComparTrainAccur(ww, delta): #对比更新前后的精确度
    global myclient
    weights = np.array(ww)
    update = np.array(delta)

    myclient.updateModel(weights)
    original = myclient.getTrainAccur()
    print("before update train accure is :", original)

    myclient.updateModel(weights + update)
    after = myclient.getTrainAccur()
    print("after update train accure is :", after)

    return after - original

def krum(deltas, clip):  #返回认为诚实训练者  #2f +2 < n
    #  deltas 是 训练者数量* 7850的二维数组
    n = len(deltas)  #人数
    deltas = np.array(deltas)
    scores = get_krum_scores(deltas, n - clip) #计算与他最近的n-clip个梯度与他距离的和
    good_idx = np.argpartition(scores, n - clip)[:(n - clip)] #先从小到大进行排序，根据索引x处的数值y，把数组中的元素划分成两半，使得index前面的元素都不大于y，
    # index后面的元素都不小于y,返回的是排序后的数据在原来数组的索引 。#取前n - clip 个si得分最低的训练者在deltas中的索引

    print(good_idx)

    return good_idx

    # return np.mean(deltas[good_idx], axis=0)

def get_krum_scores(X, groupsize):
    krum_scores = np.zeros(len(X))

    # 计算欧氏距离  #d = 根号下(A - B)*(A - B)^T ,A,B是向量
    distances = np.sum(X ** 2, axis=1)[:, None] + np.sum(  #axis = 1,行求和  #None表示维度扩充
        X ** 2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:groupsize ])

    return krum_scores

def comm(coff):  #一维浮点数数组，精度，底数
    g = 7
    accur = 2
    p = 5527
    arrFloat = np.array(coff)    #float类型
    arrInt = np.empty(len(arrFloat))
    for i in range(len(arrFloat)):
        arrInt[i] = int(arrFloat[i]*(10**accur))      #以精度e转int
    lenn = int(len(arrInt)/10)
    arrTran = np.empty([lenn,10])
    for i in range(len(arrTran)):
        for j in range(10):
            arrTran[i][j] = arrInt[i*10+j]
    res = 1
    for i in range(len(arrTran)):
        for j in range(10):
            cof = int(arrTran[i][j])  #防止runtimewaring#防止go中float转int产生进位
            temp = g**j
            num = temp**cof
            num = num % p
            res = res*num
            res = res % p

    return res
def closeWrite():
    writer.close()


def lagelangr(serect):
    x = np.zeros(len(serect))
    y = np.zeros(len(serect))
    for  i in range(len(serect)):
        if (serect[i] != 0):
            x[i] = i
            y[i] = serect[i]
    return lagrange(x, y).c


global myclient
init()

a = np.random.randn(7850)
updateModel(a)

for i in range(300):
    print("----------------itreation ", i, "------------------")
    b1 = privateFun(a)
    #b2 = privateFun(a)
 #   b3 = privateFun(a)
 #   b4 = privateFun(a)
 #   b5 = privateFun(a)
    #b = (b1 + b2 + b3 + b4 +b5 )/ 5
    myclient.simpleStep(b1)
    a = myclient.getModelWeights()
   # a = a + b*0.001

    temp = getTestAccur(a, i)
   # temp = myclient.getTestAccur()
    print("accuracy is", temp)

closeWrite()

#print('{:.3f}'.format(154.2))
#print(int(6.35))
#print(int(6.95))


# a = np.random.rand(50)
# for i in range(len(a)):
#     a[i] = '{:.2f}'.format(a[i])
# resa  = comm(a)
# print(resa)


"""
b = np.random.rand(50)
for i in range(len(a)):
    b[i] = '{:.2f}'.format(b[i])
resb = comm(b,1,1,1)
print(resb)

c = a + b
resc = comm(c,1,1,1)
print(resc)
res = (resb*resa)%5527
print(res)
print(resc == res)
"""




"""
global myclient
init()
# print(getNoise(3))
#n = myclient.getModel()
a = np.random.randn(7850)
updateModel(a)
#print("before update test accure is :", myclient.getTestAccur())
#print("before update train accure is :",myclient.getTrainAccur())
#c = myclient.getModel()
b = privateFun(a)  # 计算梯度  #验证梯度时，梯度可能都是负的？
print(b)
z = comm(b)
print(z)

simpleStep(b)
#d = myclient.getModel()

#print("after update test accure is :", myclient.getTestAccur())

#print("after update train accure is :",myclient.getTrainAccur())

print("train improve",ComparTrainAccur(a,b))

#print(d)
global myclient
init()
a = np.random.randn(7850)
updateModel(a)
for i in range(300):
    b = privateFun(a)
    c = privateFun(a)
    d = privateFun(a)
    e = b + c + d
    simpleStep(e)

    #a = myclient.getModel()

    getTestAccur(a,i)

closeWrite()
"""






