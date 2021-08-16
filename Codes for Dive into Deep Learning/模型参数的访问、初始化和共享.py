import torch
import torch.nn as nn
from torch.nn import init

net=nn.Sequential(nn.Linear(4,3),nn.ReLU(),nn.Linear(3,1))

print(type(net.parameters()))
print(type(net.named_parameters()))#访问整个网络的参数，用parameters()和named_parameters()都可以，后者除了返回参数 Tensor 外还会返回其名字
for name,param in net.named_parameters():
    print(name,param.size())

for name,param in net[0].named_parameters():#访问网络的某一层参数，用net[]来指示某一层，序号为0时为第一层
    print(name,param)
'''返回的 param 的类型为 torch.nn.parameter.Parameter ，其实这是 Tensor 的子类，和 Tensor 不同的是如果一个 Tensor 是 Parameter ，那么它会自动被添加到模型的参数列表里'''

for name,param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param,mean=0,std=0.01)
        print(name,param.data)
    if 'bias' in name:
        init.constant_(param,val=0)
        print(name,param.data)
#将网络中的权重参数用正态分布初始化，偏置参数用常数初始化
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.weight1=nn.Parameter(torch.rand(20, 20))#如果加入了nn.Parameter()，则这个定义的参数是可训练的
        self.weight2 = torch.rand(20, 20)
    def forward(self, x):
        pass
n = MyModel()
for name, param in n.named_parameters():
    print(name)#最终打印出来只有weight1没有weight2

linear=nn.Linear(1,1,bias=False)
net=nn.Sequential(linear,linear)
print(net)
for name,param in net.named_parameters():
    init.constant_(param,val=3)
    print(name,param.data)
print(id(net[0]) == id(net[1]))
print(id(net[0].weight) == id(net[1].weight))#在内存中，这两个线性层其实一个对象

x=torch.ones([1,1])
y=net(x).sum()
print(y)
y.backward()
print(net[1].weight.grad)
print(net[0].weight.grad)#因为模型参数里包含了梯度，所以在反向传播计算时，这些共享的参数的梯度是累加的.单次梯度是3，两次所以就是6
