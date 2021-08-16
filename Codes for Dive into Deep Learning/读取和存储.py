import torch
from torch import nn

x=torch.ones(3)
torch.save(x,'x.pt')#.pt是储存tensor的格式，从本机的数据集FashionMNIST也可以看出。处理后的数据是train.pt和test.pt
x2=torch.load('x.pt')
print(x2)#利用save和load储存单个tensor

y = torch.zeros(4)
torch.save([x, y], 'xy.pt')
xy_list = torch.load('xy.pt')
print(xy_list)#利用save和load储存tensor列表

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(3, 2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(2, 1)
    def forward(self, x):
        a = self.activation(self.hidden(x))
        return self.output(a)
net = MLP()
print(net.state_dict())#state_dict()可以访问模型中的可训练参数

optimizer = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
print(optimizer.state_dict())#注意，只有具有可学习参数的层(卷积层、线性层等)才有 state_dict 中的条目。优化器( optim )也有一个 state_dict ，其中包含关于优化器状态以及所使用的超参数的信息

torch.save(net.state_dict(),'path')
net1=MLP()
net1.load_state_dict(torch.load('path'))#保存和加载模型参数(state_dict())

torch.save(net,'path')
net1=torch.load('path')#保存和加载整个模型