import torch
from torch import nn

class centered_layer(nn.Module):
    def __init__(self):
        super(centered_layer, self).__init__()

    def forward(self,x):
        return x-x.mean()
#以上是一个不含模型参数的自定义层
layer=centered_layer()
print(layer(torch.tensor([1,2,3,4,5],dtype=torch.float)))


class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params =nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
#以上是一个含模型参数的自定义层
net = MyDense()
print(net)