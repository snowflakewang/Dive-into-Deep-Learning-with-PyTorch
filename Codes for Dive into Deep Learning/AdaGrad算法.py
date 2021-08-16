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

'''以下是AdaGrad算法，用一个具体函数f_2d为例'''
def adagrad_2d(x1,x2,s1,s2):
    g1,g2,eps=0.2*x1,4*x2,1e-6
    s1+=g1**2
    s2+=g2**2
    x1-=eta/math.sqrt(s1+eps)*g1
    x2-=eta/math.sqrt(s2+eps)*g2
    return (x1,x2,s1,s2)

def f_2d(x1,x2):
    return 0.1*x1**2+2*x2**2

eta=0.7
d2l.show_trace_2d(f_2d,d2l.train_2d(adagrad_2d))

'''以下是AdaGrad算法的从零开始实现，并非以某个具体函数为例，是通用算法'''
def init_adagrad_states():#后面的d2l.train_ch7()里定义了一个线性回归网络
    s_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    s_b=torch.zeros(1,dtype=torch.float32)
    return (s_w,s_b)

def adagrad(params,states,hyperparams):
    eps=1e-6
    for p,s in zip(params,states):
        s+=p.grad.data**2
        p.data-=hyperparams['lr']/torch.sqrt(s+eps)*p.grad

features,labels=get_data_ch7()
#d2l.train_ch7(adagrad,init_adagrad_states(),{'lr':0.02},features,labels)
d2l.train_pytorch_ch7(optim.Adagrad,{'lr':0.02},features,labels)
