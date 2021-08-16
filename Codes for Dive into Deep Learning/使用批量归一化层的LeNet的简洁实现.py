import time
import torch
from torch import nn,optim
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net=nn.Sequential(nn.Conv2d(1,6,5),nn.BatchNorm2d(num_features=6),nn.Sigmoid(),nn.MaxPool2d(2,2),nn.Conv2d(6,16,5),
                  nn.BatchNorm2d(num_features=16),nn.Sigmoid(),nn.MaxPool2d(2,2),d2l.FlattenLayer(),nn.Linear(16*4*4,120),
                  nn.BatchNorm1d(num_features=120),nn.Sigmoid(),nn.Linear(120,84),nn.BatchNorm1d(num_features=84),
                  nn.Sigmoid(),nn.Linear(84,10))

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)
num_epochs,lr=5,0.001
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
