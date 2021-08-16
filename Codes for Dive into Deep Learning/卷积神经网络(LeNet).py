import torch
import torchvision
import time
import sys
from torch import nn,optim
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,6,5),nn.Sigmoid(),nn.MaxPool2d(2,2),nn.Conv2d(6,16,5),nn.Sigmoid(),nn.MaxPool2d(2,2))
        self.fc=nn.Sequential(nn.Linear(16*4*4,120),nn.Sigmoid(),nn.Linear(120,84),nn.Sigmoid(),nn.Linear(84,10))

    def forward(self,img):
        feature=self.conv(img)
        output=self.fc(feature.view(img.shape[0],-1))
        return output

batch_size=256
net=LeNet()

def load_data_fashion_mnist(batch_size, resize=None, root='D:/新建文件夹/Pycharm Community/datasets'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=False, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=False, transform=transform)
    if sys.platform.startswith('win'):  # 如果用的是windows系统
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter

def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n

def train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs):
    net=net.to(device)
    print('training on',device)
    loss=torch.nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,batch_count,start=0.0,0.0,0,0,time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
            test_acc=evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc% .3f, testacc % .3f, time % .1fsec'% (epoch + 1, train_l_sum /batch_count, train_acc_sum / n, test_acc,time.time() - start))

train_iter,test_iter=load_data_fashion_mnist(batch_size=batch_size)

lr,num_epochs=0.001,5
print(net.parameters())
optimizer=optim.Adam(net.parameters(),lr)
train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)
'''到这里为止第一阶段的coding相关任务貌似就结束了，目前已经可以利用Pytorch搭建LeNet-5、AlexNet、VGGNet这样的框架，后面可以再搭建一下ResNet和DenseNet，这样就完成了Computer Vision领域
比较经典的几个网络的搭建。不过这样的程度还是远远不够，还有大量值得深入的地方。可以先以LeNet-5为例，把整个搭建网络的过程再走一遍，有一些调用现成库函数的代码值得深挖一下'''

'''关于optimizer.zero_grad()，l.backward()，optimizer.step()
optimizer.zero_grad()的作用是将梯度清零，l.backward()的作用是计算每一个参数（权重和偏置），optimizer.step()的作用是进行梯度下降进行一步参数更新。

l.backward():
PyTorch的反向传播(即tensor.backward())是通过autograd包来实现的，autograd包会根据tensor进行过的数学运算来自动计算其对应的梯度。
具体来说，torch.tensor是autograd包的基础类，如果你设置tensor的requires_grads为True，就会开始跟踪这个tensor上面的所有运算，如果你做完运算后使用tensor.backward()，所有的梯度就会自动运算，tensor的梯度将会累加到它的.grad属性里面去。
更具体地说，损失函数loss是由模型的所有权重经过一系列运算得到的，若某个权重的requires_grads为True，则w的所有上层参数（后面层的权重w）的.grad_fn属性中就保存了对应的运算，然后在使用loss.backward()后，会一层层的反向传播计算每个w的梯度值，并保存到该w的.grad属性中。

optimizer.step():
step()函数的作用是执行一次优化步骤，通过梯度下降法来更新参数的值。因为梯度下降是基于梯度的，所以在执行optimizer.step()函数前应先执行loss.backward()函数来计算梯度。
注意：optimizer只负责通过梯度下降进行优化，而不负责产生梯度，梯度是.backward()方法产生的。'''