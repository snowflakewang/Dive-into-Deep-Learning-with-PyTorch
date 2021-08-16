import torch
from torch import nn,optim
import torch.nn.functional as f
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv=False,stride=1):
        super(residual, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)

    def forward(self,X):
        Y=f.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        return f.relu(Y+X)

def resnet_block(in_channels,out_channels,num_residuals,first_block=False):
    if first_block:
        assert in_channels==out_channels
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(residual(in_channels,out_channels,use_1x1conv=True,stride=2))
        else:
            blk.append(residual(out_channels,out_channels))
    return nn.Sequential(*blk)

net=nn.Sequential(nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),nn.BatchNorm2d(64),nn.ReLU(),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

net.add_module('resnet_block1',resnet_block(64,64,2,first_block=True))
net.add_module('resnet_block2',resnet_block(64,128,2))
net.add_module('resnet_block3',resnet_block(128,256,2))
net.add_module('resnet_block4',resnet_block(256,512,2))
net.add_module('global_avg_pool',d2l.GlobalAvgPool2d())
net.add_module('fc',nn.Sequential(d2l.FlattenLayer(),nn.Linear(512,10)))

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=96)
num_epochs,lr=5,0.001
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)