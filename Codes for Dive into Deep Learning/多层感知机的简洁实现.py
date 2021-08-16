import torch
from torch import nn,optim
import d2lzh_pytorch.utils as d2l

batch_size=256
num_inputs,num_outputs,num_hiddens=28*28,10,256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

net=nn.Sequential(d2l.FlattenLayer(),nn.Linear(num_inputs,num_hiddens),nn.ReLU(),nn.Linear(num_hiddens,num_outputs))
for param in net.parameters():
    nn.init.normal_(param,mean=0,std=0.01)

loss=nn.CrossEntropyLoss()

def train_ch3(net,data_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y).sum()

            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum+=l.item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().item()
            n+=y.shape[0]
        test_acc=d2l.evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train_acc %.3f,test_acc %.3f'%(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc))

num_epochs,lr=10,0.5
optimizer=optim.Adadelta(net.parameters(),lr)
train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)