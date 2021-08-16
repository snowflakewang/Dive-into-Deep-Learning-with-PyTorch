import d2lzh_pytorch.utils as d2l
import numpy as np
import torch
from torch import nn,optim

def get_data_ch7():
    data = np.genfromtxt('D:/新建文件夹/Pycharm Community/动手学深度学习/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

'''以下是梯度下降算法，用一个具体函数f_2d为例，为了与后面带动量的梯度下降算法做对比'''
def f_2d(x1,x2):
    return 0.1*x1**2+2*x2**2

def gd_2d(x1,x2,s1,s2):
    return (x1-eta*0.2*x1,x2-eta*4*x2,0,0)

eta=0.6#当学习率设置为0.6时就会发散
d2l.show_trace_2d(f_2d,d2l.train_2d(gd_2d))

'''以下是带动量的梯度下降算法，用一个具体函数f_2d为例'''
def momentum_2d(x1,x2,v1,v2):
    v1=gamma*v1+eta*0.2*x1
    v2=gamma*v2+eta*4*x2
    return (x1-v1,x2-v2,0,0)

eta,gamma=0.4,0.5
d2l.show_trace_2d(f_2d,d2l.train_2d(momentum_2d))

'''以下是带动量的梯度下降算法的从零开始实现，并非以某个具体函数为例，是通用算法'''
features,labels=get_data_ch7()

def init_momentum_states():
    v_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    v_b=torch.zeros(1,dtype=torch.float32)
    return (v_w,v_b)

def sgd_momentum(params,states,hyperparams):
    for p,v in zip(params,states):
        v.data=hyperparams['momentum']*v.data+hyperparams['lr']*p.grad.data
        p.data-=v.data

#d2l.train_ch7(sgd_momentum,init_momentum_states(),{'momentum':0.5,'lr':0.02},features,labels)
d2l.train_pytorch_ch7(optim.SGD,{'momentum':0.5,'lr':0.02},features,labels)