'''this file is related to Autograd:自动求梯度 in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
import numpy as np

x=torch.ones(2,2,requires_grad=True)
y=x+2
print(y,y.grad_fn)
z=y*y*3
out=z.mean()
out.backward()
print(x.grad)
