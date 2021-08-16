import d2lzh_pytorch.utils as d2l
import torch
from torch import optim
import numpy as np

def get_data_ch7():
    data = np.genfromtxt('D:/新建文件夹/Pycharm Community/动手学深度学习/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

'''以下是AdaDelta算法的从零开始实现，并非以某个具体函数为例，是通用算法'''
def init_adadelta_states():
    s_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    s_b=torch.zeros(1,dtype=torch.float32)
    delta_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    delta_b=torch.zeros(1,dtype=torch.float32)
    return ((s_w,delta_w),(s_b,delta_b))

def adadelta(params,states,hyperparams):
    rho,eps=hyperparams['rho'],1e-6
    for p,(s,delta) in zip(params,states):
        s.data=rho*s.data+(1.0-rho)*p.grad.data**2#s里包含s_w和s_b,这两者的更新算法是相同的
        g=p.grad.data*torch.sqrt((delta.data+eps)/(s.data+eps))
        p.data-=g
        delta.data=rho*delta.data+(1.0-rho)*g**2#delta里包含delta_w和delta_b,这两者的更新算法是相同的

features,labels=get_data_ch7()
#d2l.train_ch7(adadelta,init_adadelta_states(),{'rho':0.9},features,labels)
d2l.train_pytorch_ch7(optim.Adadelta,{'rho':0.9},features,labels)