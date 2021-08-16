'''this file is related to 深度学习基础：线性回归 in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
import numpy as np
import random

num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2#设定该学习算法的预期结果，即真值
features=torch.randn(num_examples,num_inputs,dtype=torch.float32)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b#这个得到的labels是真值
labels+=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float32)#利用高斯分布引入一些噪声然后加到原有的真值labels上，此时的labels可以视为训练用数据集
#以上人工生成数据集的过程

def data_iter(batch_size,features,labels):
    num_examples=len(features)
    indices=list(range(num_examples))
    random.shuffle(indices)#样本读取顺序随机，random.shuffle直接在indices原数组上做随机打乱
    for i in range(0,num_examples,batch_size):#按照batch_size的步长来自增i
        j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])#最后一个batch可能达不到batch_size的大小，因此min(i+batch_size,num_examples)用来防止读取最后一个batch时溢出
        yield features.index_select(0,j),labels.index_select(0,j)
        #yield和return有相似的地方，当函数运行到yield和return所在的地方时，它们都会使函数返回指定的值。但是，如果在以后该函数再次被调用，含有return的函数就会从函数的头部开始重新运行，而含有yield的函数就会从上一次yield退出的地方继续运行
        #在这里，每进行一次for loop，都会退出函数返回一个batch的数据，然后下一次进入函数时，由于yield的特性，for loop会一直向前进行而不是重新从一开始循环
        #这里的j是一个tensor，是一个1*min(batch_size,num_examples-i)的tensor，则j=torch.LongTensor(indices[i:min(i+batch_size,num_examples)])是本次batch要在features，labels抽取出的行的序号（因为一行是一个样本）

batch_size=10
'''for X, y in data_iter(batch_size, features,labels):
    print(X, y)
    break'''
#以上是读取数据集的过程

w=torch.randn((num_inputs,1),dtype=torch.float32)#还可以用w=torch.tensor(np.random.normal(0,0.01,(num_inputs,1)),dtype=torch.float32)进行权重的高斯分布初始化
b=torch.zeros(1,dtype=torch.float32)

w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#以上是训练参数的初始化

def linreg(X,w,b):
    return torch.mm(X,w)+b#mm函数用于计算矩阵乘法
#以上是模型定义，本次定义的是一个线性函数，即学习算法最终学到的也是一个线性函数

def squared_loss(y_hat,y):
    return (y_hat-y.view(y_hat.size()))**2/2#返回值和y_hat是同型的tensor
#以上是定义损失函数，使用的是均方误差损失函数

def mbsgd(params,lr,batch_size):
    for param in params:
        param.data-=lr*param.grad/batch_size#注意这里改变param时用的是param.data
#以上是优化算法，用的是mini-batch stochastic gradient descent算法

lr=0.03
num_epochs=5
net=linreg#注意这里的linreg定义成了一个函数而非一个类，因此写成net=linreg而非net=linreg()
loss=squared_loss#与上同理
for epoch in range(num_epochs):
    for X,y in data_iter(batch_size,features,labels):#除去最后一个可能的形状特殊的batch，X是一个batch_size*2的tensor，y是一个batch_size*1的tensor，
        #因此net(X,w,b),y)=torch.mm(X,w)+b中，mm(X,w)是一个batch_size*2和一个2*1的tensor矩阵相乘，得到batch_size*1的tensor，再利用broadcast机制与b相加，得到的y_hat是一个batch_size*1的tensor
        l=loss(net(X,w,b),y).sum()#loss(net(X,w,b),y)返回一个batch_size*1的tensor，每一元素都是一个样本的(y_hat-y)**2/2,
        #考虑到平方误差的表达式为sigma(1<=i<=batch_size) (y_hat_i-y_i)**2/2/batch_size，因此应该把loss(net(X,w,b),y)的所有维度加起来，
        #这样也就把l转化成了一个scalar，也就可以用backward()求导了
        l.backward()#这一步是反向传播，只是求出了l关于每一个权重和偏置的导数，但还没有更新参数，注意区分反向传播和梯度下降
        mbsgd([w,b],lr,batch_size)#这一步才是梯度下降，才是更新参数的过程
        w.grad.data.zero_()
        b.grad.data.zero_()#不要忘记梯度清零，因为l.backward()算出的梯度会叠加到原来的已经算出的梯度上
    train_l = loss(net(features, w, b), labels)#在一个epoch结束后，用目前训练得到的w和b与features相乘得到预测结果，把这个结果与真实结果对比算出到目前为止的损失函数大小，
    #此时train_l是一个num_examples*1的tensor，对它做mean()（也就是先相加再取平均）然后再用item()把它转化成标量输出
    print('epoch %d, loss %f' % (epoch + 1,train_l.mean().item()))
#以上是模型训练

print(true_w, '\n', w)
print(true_b, '\n', b)

