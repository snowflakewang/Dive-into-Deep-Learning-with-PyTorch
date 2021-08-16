import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden=nn.Linear(784,256)
        self.activation=nn.ReLU()
        self.output=nn.Linear(256,10)

    def forward(self,x):
        a=self.activation(self.hidden(x))
        return self.output(a)
    #以上的 MLP 类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的backward函数

'''X=torch.rand([2,784])
print(X)
net=MLP()
print(net)
print(net(X))'''

class FancyMLP(nn.Module):
    def __init__(self):
        super(FancyMLP,self).__init__()

        self.rand_weight=torch.rand([20,20],requires_grad=False)#不可训练参数，即常数参数
        self.linear=nn.Linear(20,20)

    def forward(self,x):
        x=self.linear(x)
        x=nn.functional.relu(torch.mm(x,self.rand_weight.data)+1)
        x=self.linear(x)# 与第30行复用全连接层。等价于两个全连接层共享参数
        print(x.norm().item())
        sum = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                sum+=x[i][j]**2
        print(math.sqrt(sum))
        while x.norm().item() > 1:#norm()的含义是求范数，这里应该默认是L2范数，在本行的上方写了一个验证程序来验证
            x /= 2
        if x.norm().item() < 0.8:
            x *= 10
        return x.sum()

'''Sequential 、 ModuleList 、 ModuleDict 类都继承自 Module 类。
与 Sequential 不同， ModuleList 和 ModuleDict 并没有定义一个完整的网络，它们只是将不同的模块存放在一起，需要自己定义 forward 函数，而Sequential不用定义forward函数。
虽然 Sequential 等类可以使模型构造更加简单，但直接继承 Module 类可以极大地拓展模型构造的灵活性。'''
mlp=FancyMLP()
x=torch.ones((1,20))
mlp(x)