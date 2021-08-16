import time
import torch
from torch import nn,optim
import d2lzh_pytorch.utils as d2l

def batch_norm(is_training,X,gamma,beta,moving_mean,moving_var,eps,momentum):
    if not is_training:
        X_hat=(X-moving_mean)/torch.sqrt(moving_var+eps)
    else:
        assert len(X.shape) in (2,4)
        if len(X.shape)==2:
            mean=X.mean(dim=0)
            var=((X-mean)**2).mean(dim=0)
        else:
            mean=X.mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
            var=((X-mean)**2).mean(dim=0,keepdim=True).mean(dim=2,keepdim=True).mean(dim=3,keepdim=True)
        X_hat=(X-mean)/torch.sqrt(var+eps)
        moving_mean=momentum*moving_mean+(1.0-momentum)*mean
        moving_var=momentum*moving_var+(1.0-momentum)*var#这两式均为滑动平均公式
    Y=gamma*X_hat+beta#拉伸和偏移
    return Y,moving_mean,moving_var

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):#num_features对于卷积层来说是输出通道数，对于全连接层来说是输出个数，num_dims对于全连接层来说是2，对卷积层是4
        super(BatchNorm, self).__init__()
        if num_dims==2:
            shape=(1,num_features)
        else:
            shape=(1,num_features,1,1)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
        self.moving_mean=torch.zeros(shape)
        self.moving_var=torch.zeros(shape)

    def forward(self,X):
        if self.moving_mean.device!=X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var=self.moving_var.to(X.device)
        Y,self.moving_mean,self.moving_var=batch_norm(self.training,X,self.gamma,self.beta,self.moving_mean,self.moving_var,eps=1e-5,momentum=0.9)
        return Y