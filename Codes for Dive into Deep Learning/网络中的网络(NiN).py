import torch
from torch import nn,optim
import torch.nn.functional as f
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def nin_block(in_channels,out_channels,kernel_size,stride,padding):
    blk=nn.Sequential(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),nn.ReLU(),nn.Conv2d(out_channels,out_channels,1),
                      nn.ReLU(),nn.Conv2d(out_channels,out_channels,1),nn.ReLU())
    return blk

class global_avg_pool2d(nn.Module):
    def __init__(self):
        super(global_avg_pool2d, self).__init__()
    def forward(self,x):
        return f.avg_pool2d(x,kernel_size=x.size()[2:])

net=nn.Sequential(nin_block(1,96,kernel_size=11,stride=4,padding=0),nn.MaxPool2d(kernel_size=3,stride=2),nin_block(96,256,kernel_size=5,stride=1,padding=2),
                  nn.MaxPool2d(kernel_size=3,stride=2),nin_block(256,384,kernel_size=3,stride=1,padding=1),nn.MaxPool2d(kernel_size=3,stride=2),
                  nn.Dropout(0.5),nin_block(384,10,kernel_size=3,stride=1,padding=1),global_avg_pool2d(),d2l.FlattenLayer())

batch_size=128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)

num_epochs,lr=5,0.002
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)