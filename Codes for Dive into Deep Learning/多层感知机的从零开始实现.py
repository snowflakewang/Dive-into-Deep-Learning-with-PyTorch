import torch
from torch import nn
import numpy as np
import d2lzh_pytorch.utils as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs,num_outputs,num_hiddens=28*28,10,256
w1=torch.randn((num_inputs,num_hiddens),dtype=torch.float32)
b1=torch.zeros(num_hiddens,dtype=torch.float32)
w2=torch.randn((num_hiddens,num_outputs),dtype=torch.float32)
b2=torch.zeros(num_outputs,dtype=torch.float32)
params=[w1,b1,w2,b2]
for param in params:
    param.requires_grad=True

def relu(X):
    return torch.max(X,other=torch.tensor(0.0))

def net(X):
    X=X.view((-1,num_inputs))
    output1=torch.mm(X,w1)+b1
    act1=relu(output1)
    output2=torch.mm(act1,w2)+b2
    return output2

loss=nn.CrossEntropyLoss()

num_epochs,lr=10,100.0
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)