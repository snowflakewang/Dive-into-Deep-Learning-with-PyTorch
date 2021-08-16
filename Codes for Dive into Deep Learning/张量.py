'''this file is related to Tensors in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
import numpy as np

x=torch.empty(5,3,2)
print(x)
y=torch.rand(3,3,2)
print(y)
x=torch.rand(3,3,2)
print(x)
print(x+y)
y=x[0]
print(y)
svd_ans=torch.svd(y)
print(svd_ans)
a=torch.ones(4,2)
print(a)
b=a.numpy()
print(b)
a=np.ones(4)
print(a)
b=torch.from_numpy(a)
print(b)
if torch.cuda.is_available():
    device=torch.device("cuda")

    y=torch.ones_like(x,device=device)#直接创建一个GPU上的tensor
    #ones_like和zeros_like的功能是生成一个和传入tensor x尺寸相同的tensor
    x=x.to(device)#把x从CPU移动到GPU上
    z=x+y
    print(z)
