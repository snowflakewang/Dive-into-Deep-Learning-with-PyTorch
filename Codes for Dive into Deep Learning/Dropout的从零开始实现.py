import torch
import numpy as np
import d2lzh_pytorch.utils as d2l

def dropout(X,drop_prob):
    X=X.float()
    assert 0<=drop_prob<=1
    #这是断言函数，如果断言函数的条件成立，不会触发任何操作；但如果不满足，程序就会直接报错并且返回AssertionError
    keep_prob=1-drop_prob
    if keep_prob==0:
        return torch.zeros_like(X)
    mask=(torch.rand(X.size())<keep_prob).float()
    #torch.rand()函数指的是从[0,1)的均匀分布中抽取随机数，这刚好可以作为一个概率值来使用
    #print(mask)#mask是一个由0和1组成的tensor，每个位置取0服从二项分布B(1,drop_prob)，反之取1
    return mask*X/keep_prob#做逐个元素点乘
'''X=torch.tensor([[1,2,3],[4,5,6]])
a=dropout(X,0.3)
print(a)'''

num_inputs,num_outputs,num_hiddens1,num_hiddens2=784,10,256,256
drop_prob1,drop_prob2=0.2,0.5
w1=torch.tensor(np.random.normal(0,0.01,[num_inputs,num_hiddens1]),dtype=torch.float,requires_grad=True)
b1=torch.zeros(num_hiddens1,requires_grad=True)
w2=torch.tensor(np.random.normal(0,0.01,[num_hiddens1,num_hiddens2]),dtype=torch.float,requires_grad=True)
b2=torch.zeros(num_hiddens2,requires_grad=True)
w3=torch.tensor(np.random.normal(0,0.01,[num_hiddens2,num_outputs]),dtype=torch.float,requires_grad=True)
b3=torch.zeros(num_outputs,requires_grad=True)
params=[w1,b1,w2,b2,w3,b3]

def net(X,is_training=True):
    X=X.view(-1,num_inputs)
    H1=(torch.matmul(X,w1)+b1).relu()
    if is_training:
        H1=dropout(H1,drop_prob1)
    H2=(torch.matmul(H1,w2)+b2).relu()
    if is_training:
        H2=dropout(H2,drop_prob2)
    return torch.matmul(H2,w3)+b3

num_epochs,lr,batch_size=5,100.0,256
loss=torch.nn.CrossEntropyLoss()
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
