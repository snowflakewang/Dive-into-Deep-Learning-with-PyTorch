import d2lzh_pytorch.utils as d2l
import torch
from torch import optim
import numpy as np

def get_data_ch7():
    data = np.genfromtxt('D:/新建文件夹/Pycharm Community/动手学深度学习/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

'''以下是Adam算法的从零开始实现，并非以某个具体函数为例，是通用算法'''
def init_adam_states():
    v_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    v_b=torch.zeros(1,dtype=torch.float32)
    s_w=torch.zeros((features.shape[1],1),dtype=torch.float32)
    s_b=torch.zeros(1,dtype=torch.float32)
    return ((v_w,s_w),(v_b,s_b))

def adam(params,states,hyperparams):
    beta1,beta2,eps=0.9,0.999,1e-6
    for p,(v,s) in zip(params,states):
        v=beta1*v+(1.0-beta1)*p.grad.data
        s=beta2*s+(1.0-beta2)*p.grad.data**2
        v_bias_corr=v/(1.0-beta1**hyperparams['t'])
        s_bias_corr=s/(1.0-beta2**hyperparams['t'])
        g=hyperparams['lr']*v_bias_corr/torch.sqrt(s_bias_corr+eps)
        p.data-=g
        hyperparams['t']+=1

features,labels=get_data_ch7()
#d2l.train_ch7(adam,init_adam_states(),{'lr':0.01,'t':1},features,labels)
d2l.train_pytorch_ch7(optim.Adam,{'lr':0.01},features,labels)