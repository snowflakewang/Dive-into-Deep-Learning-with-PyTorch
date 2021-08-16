'''this file is related to softmax回归的简洁实现 in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
from torch import nn
from torch.nn import init
import d2lzh_pytorch.utils as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
#获取和读取数据

num_inputs=28*28
num_outputs=10

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.Linear=nn.Linear(num_inputs,num_outputs)

    def forward(self,x):#x.shape:(batch_size,1,28,28)
        y=self.Linear(x.view(x.shape[0],-1))
        return y

net=LinearNet(num_inputs,num_outputs)
init.normal_(net.Linear.weight,mean=0,std=0.01)
init.constant_(net.Linear.bias,val=0)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(net.parameters(),lr=0.1)
num_epochs = 5

d2l.train_ch3(net, train_iter, test_iter, loss,num_epochs, batch_size, None, None, optimizer)
#d2l模块中数据集的加载路径是C:/用户/Wang Yitong/Datasets,并且设置为download=False，因此在C目录下应放置一个FashionMNIST数据集