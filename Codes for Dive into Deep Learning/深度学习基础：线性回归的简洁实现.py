'''this file is related to 深度学习基础：线性回归的简洁实现 in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
from torch import nn
import numpy as np
import torch.utils.data as data#PyTorch提供了 data 包来读取数据

num_inputs=2
num_examples=1000
true_w=[2,-3.4]
true_b=4.2
features=torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype=torch.float)
labels=true_w[0]*features[:,0]+true_w[1]*features[:,1]+true_b#这个得到的labels是真值
labels+=torch.tensor(np.random.normal(0,0.01,labels.size()),dtype=torch.float)#利用高斯分布引入一些噪声然后加到原有的真值labels上，此时的labels可以视为训练用数据集
#以上是人工生成的数据集

batch_size=10
dataset=data.TensorDataset(features,labels)
data_iter=data.DataLoader(dataset,batch_size,shuffle=True)#此处data_iter的作用与深度学习基础：线性回归.py中的相同
#以上是读取数据，此处直接调用了torch中的现成函数

class linear_net(nn.Module):
    def __init__(self,n_feature):
        super(linear_net,self).__init__()
        self.linear=nn.Linear(n_feature,1)

    def forward(self,x):#forward定义了前向传播
        y=self.linear(x)
        return y

net=linear_net(num_inputs)#num_inputs对应n_feature
#事实上我们还可以用 nn.Sequential 来更加方便地搭建网络，Sequential是一个有序的容器，网络层将按照在传入Sequential的顺序依次被添加到计算图中
# 写法一
'''net = nn.Sequential(
 nn.Linear(num_inputs, 1)
 # 此处还可以传入其他层
 )
# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs,
1))
# net.add_module ......
# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
 ('linear', nn.Linear(num_inputs, 1))
 # ......
 ]))'''
#可以通过 net.parameters() 来查看模型所有的可学习参数，此函数将返回一个生成器
'''for param in net.parameters():
    print(param)'''
#以上是模型定义，本次定义的是一个线性函数，即学习算法最终学到的也是一个线性函数

from torch.nn import init

init.normal_(net.linear.weight,mean=0,std=0.01)
init.constant_(net.linear.bias,val=0)
#以上是可训练参数的初始化

loss=nn.MSELoss()
#以上是定义损失函数，采用均方误差损失函数

import torch.optim as optim

optimizer=optim.Adam(net.parameters(),lr=0.03)
#我们还可以为不同子网络设置不同的学习率，这在finetune时经常用到
'''optimizer =optim.SGD([
 # 如果对某个参数不指定学习率，就使用最外层的默认学习率
 {'params':
net.subnet1.parameters()}, # lr=0.03
 {'params':
net.subnet2.parameters(), 'lr': 0.01}
 ], lr=0.03)'''
#以上是定义优化算法

num_epochs=10
for epoch in range(0,num_epochs):
    for X,y in data_iter:
        output=net(X)
        l=loss(output,y.view(-1,1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch+1,l.item()))
#以上是模型训练

print(true_w,net.linear.weight.data)
print(true_b,net.linear.bias.data)