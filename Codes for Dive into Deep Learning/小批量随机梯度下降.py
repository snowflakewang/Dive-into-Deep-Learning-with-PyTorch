import numpy as np
import torch
import d2lzh_pytorch.utils as d2l

def get_data_ch7():
    data = np.genfromtxt('D:/新建文件夹/Pycharm Community/动手学深度学习/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return torch.tensor(data[:1500, :-1], dtype=torch.float32), \
        torch.tensor(data[:1500, -1], dtype=torch.float32) # 前1500个样本(每个样本5个特征)

def sgd(params,states,hyperparams):
    for p in params:
        p.data-=hyperparams['lr']*p.grad.data

def train_sgd(lr,batch_size,num_epochs=2):
    d2l.train_ch7(sgd,None,{'lr':lr},features,labels,batch_size,num_epochs)

features,labels=get_data_ch7()
print(features.shape)
train_sgd(1,1500,6)