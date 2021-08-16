import math
import d2lzh_pytorch.utils as d2l
import torch
from torch import optim
import numpy as np

def get_data_ch7():
    data = np.genfromtxt('D:/新建文件夹/Pycharm Community/动手学深度学习/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

'''以下是RMSProp算法，用一个具体函数f_2d为例'''
def f_2d(x1,x2):
    return 0.1*x1**2+2*x2**2

def rmsprop_2d(x1,x2,s1,s2):
    g1,g2,eps=0.2*x1,4*x2,1e-6
    s1=gamma*s1+(1.0-gamma)*g1**2
    s2=gamma*s2+(1.0-gamma)*g2**2
    x1-=eta/math.sqrt(s1+eps)*g1
    x2-=eta/math.sqrt(s2+eps)*g2
    return (x1,x2,s1,s2)

eta,gamma=0.4,0.9
d2l.show_trace_2d(f_2d,d2l.train_2d(rmsprop_2d))

'''以下是RMSProp算法的从零开始实现，并非以某个具体函数为例，是通用算法'''
def init_rmsprop_states():
    s_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    s_b=torch.zeros(1,dtype=torch.float32)
    return (s_w,s_b)

def rmsprop(params,states,hyperparams):
    eps=1e-6
    for p,s in zip(params,states):
        s.data=hyperparams['gamma']*s.data+(1.0-hyperparams['gamma'])*p.grad.data**2
        p.data-=hyperparams['lr']/torch.sqrt(s.data+eps)*p.grad.data
    print(p,s)

features,labels=get_data_ch7()
d2l.train_ch7(rmsprop,init_rmsprop_states(),{'lr':0.01,'gamma':0.9},features,labels)
#d2l.train_pytorch_ch7(optim.RMSprop,{'lr':0.01,'alpha':0.9},features,labels)