import torch
import numpy as np
# array1=np.arange(12).reshape(3,4)
# print(array1)
# print(torch.tensor(array1))
# print(torch.tensor(np.array([[1,2,3],[4,5,6]])))
# print(torch.randint(0,10,(3,4)))
# print(torch.randint(low=0,high=10,size=(3,4)))
# print(torch.Tensor([1,2]))
# t4=torch.tensor(np.arange(24).reshape(2,3,4)) #**生成一个三维数组，形状见输出
# print(t4)
# print(t4.transpose(0,1))#n维数组需要2个参数，作用是交换两个维度，这里使t4数组的第一维和第二维转置
# print(t4.permute(0,2,1))#n维数组，需要n个参数，把需要交换的维度标清楚
# print(t4) #这里可见转置操作不会改变张量本身的形状
# t5=torch.tensor(np.arange(15).reshape(3,5))
# print(t5)
# print(t5.t())#只支持2维 转置矩阵
# print(t5.transpose(0,1))#2维的话参数不可以省略
# print(t5.permute(1,0))#2维的话参数 不可以省略

# t4=torch.tensor(np.arange(24).reshape(2,3,4))#生成一个三维数组，形状见输出
# print(t4)
# print(t4[0,0,0])#获取数字0
# print(t4[1,2,3])#获取数字23
# print(t4[0])#与下面等价，切片
# print(t4[0,:,:])
# print(t4[:,:,1])

# print(torch.cuda.is_available())





# import torch
# a=torch.randn(2,2)
# print(a)
# a=((a*3)/(a-1))
# print(a)
# print(a.requires_grad)
# a.requires_grad=True
# # a.requires_grad(True)#**语法错误
# print(a.requires_grad)
# b=(a*a).sum()
# print(b)
# print(b.grad_fn)
# with torch.no_grad():
#     c=(a*a).sum()
#     print(c.requires_grad)

# x=torch.ones(2,2,requires_grad=True)
# print(x)
# y=x+2
# print(y)
# z=y*y*3
# print(z)
# out=z.mean()
# print(out)
# out.backward()#不可省略，使用了backward方法之后才可使用x.grad
# print(x.grad)
#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import numpy as np
from matplotlib import pyplot as plt

#准备x，y的数值
x = torch.rand([50,1])#50行一列，x相当于采样点，在0-1间随机选50个点，越多越准确
y_true = 3*x + 0.8 #按照计算公式y的准确值

#通过模型计算y_predict
w = torch.rand(1,requires_grad=True)#随机产生一个w，并追踪梯度
b = torch.rand(1,requires_grad=True)##随机产生一个b，并追踪梯度
# y_predict = torch.matmul(x,w)+b#做乘法，相当于x*w+b

#计算损失，损失函数也可以替换成其他的损失函数
def loss_fn(y_true,y_predict):
    loss=(y_predict-y_true).pow(2).mean()#做差，平方，再取平均值
    for i in [w,b]:#每次梯度都要归零
        if i.grad is not None:
            i.grad.data.zero_()#加下划线就地修改
    loss.backward() #反向传播
    return loss.data

#优化w和b的值，使之接近真实数据3和0.8
def optimize(learning_rate):
    w.data = w.data - learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data

#通过循环，反向传播，更新参数
for i in range(3000):
   # #计算预测值
    y_predict = x * w + b
    #计算损失，把参数的梯度置为0，进行反向传播
    loss=loss_fn(y_true,y_predict)

    if i%500 == 0:
        print(i,loss.data,w.data,b.data)
    #更新参数w和b
    optimize(0.01)

#绘制图像，观测训练结束的预测值和真实值
predict=x*w + b
#plt绘制图像的数据需要是numpy类型的，不能是torch类型
plt.plot(x.numpy(),y_true.numpy(),c='r') #以点阵图的形式绘制
plt.scatter(x.numpy(),predict.detach().numpy())#以直线绘制，y_predeict是带有梯度的torch类型，所以需要使用.detach()方法或者.data属性获得不含梯度的张量数据
plt.show()
