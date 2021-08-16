import torch
from torch import nn
import d2lzh_pytorch.utils as d2l

def corr2d_multi_in(X,K):
    res=d2l.corr2d(X[0,:,:],K[0,:,:])
    for i in range(1,X.shape[0]):
        res+=d2l.corr2d(X[i,:,:],K[i,:,:])
    return res

def corr2d_multi_in_out(X,K):
    return torch.stack([corr2d_multi_in(X,k)] for k in K)

def cprr2d_multi_in_out_1x1(X,K):
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.view(c_i,h*w)
    K=K.view(c_o,c_i)
    Y=torch.mm(K,X)
    return Y.view(c_o,h,w)
