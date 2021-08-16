import torch
from torch import nn

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name())

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x=torch.tensor([1,2,3],device=device)
print(x)
y=x**2
print(y)#存放在gpu上的tensor做计算的结果仍然在gpu上

net=nn.Linear(3,1)
print(list(net.parameters())[0].device)
net=net.cuda()
print(list(net.parameters())[0].device)
x=torch.rand([2,3]).cuda()#保证模型输入的 Tensor 和模型都在同一设备上
print(net(x))