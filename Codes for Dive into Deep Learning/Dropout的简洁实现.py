import torch
import torch.nn as nn
import d2lzh_pytorch.utils as d2l

num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
drop_prob1,drop_prob2=0.2,0.5
net=nn.Sequential(d2l.FlattenLayer(),nn.Linear(num_inputs,num_hiddens1),nn.ReLU(),nn.Dropout(drop_prob1),
                  nn.Linear(num_hiddens1,num_hiddens2),nn.ReLU(),nn.Dropout(drop_prob2),nn.Linear(num_hiddens2,num_outputs))

for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)

num_epochs,lr,batch_size=5,0.5,256
loss=torch.nn.CrossEntropyLoss()
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
optimizer=torch.optim.SGD(net.parameters(),lr)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)