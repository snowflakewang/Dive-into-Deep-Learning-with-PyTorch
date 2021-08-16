'''this file is related to softmax回归的从零开始实现 in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
import numpy as np
import d2lzh_pytorch.utils as d2l

batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)

num_inputs=28*28#输入图像是28*28的，因此将其拉长成向量时，该向量维度是28*28=784
num_outputs=10#因为总共有10种类别，因此分类器应该有10种输出
w=torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)#或者可以写为w=torch.randn((num_inputs,num_outputs),dtype=torch.float)
#这相当于简洁实现中的init.normal_(net.Linear.weight,mean=0,std=0.01)
b=torch.zeros(num_outputs,dtype=torch.float)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

def softmax(X):
    X_exp=X.exp()
    partition=X_exp.sum(dim=1,keepdim=True)#dim=0代表对同一列的元素求和，dim=1代表对同一行的元素求和，keepdim=True代表将求和结果保存在
    return X_exp/partition#这里应用了广播机制(broadcast)，把partition扩展成X_exp的形状

def net(X):
    return softmax(torch.mm(X.view(-1,num_inputs),w)+b)#该网络仍然采用线性模型，只是在输出层采用softmax分类输出。X是一个1*784向量，w是一个784*10矩阵，两者乘积为1*10向量，这刚好与已经定义的softmax函数里按行求和相对应

def cross_entropy(y,y_hat):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))

'''对gather函数的理解：
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3,0.2, 0.5]])
y = torch.LongTensor([0, 2])
print(y_hat.gather(1, y.view(-1, 1)))
以上代码中，y_hat是一个2*3的矩阵，y是一个1*2的向量，y.view(-1,1)是一个2*1的向量
在gather函数中，1代表按行收集数值，y.view(-1,1)给出每一行被收集的数值的索引
0.1的索引是[0][0]，0.5的索引是[1][2]，在按行收集的情况下，由于y.view(-1,1)=[[0], [2]]，
因此第一行的[0]和第二行的[2]会被收集，即第一行的0.1和第二行的0.5将被收集'''

def accuracy(y,y_hat):
    return (y_hat.argmax(dim=1)==y).float().mean().item()#这个算出来就是一个介于0，1之间的值，可以认为是一个batch上的准确率
'''其中 y_hat.argmax(dim=1) 返回矩阵 y_hat 每行中最大元素的索引，且返回结果与变量 y 形状相同。
相等条件判断式 (y_hat.argmax(dim=1) == y) 是一个类型
为 ByteTensor 的 Tensor ，我们用 float() 将其转换为
值为0（相等为假）或1（相等为真）的浮点型 Tensor'''

def evaluate_accuracy(data_iter,net):
    acc_sum,n=0,0
    for X,y in data_iter:
        acc_sum+=accuracy(y,net(X))
        n+=1#n是已经循环过的batch的数量，每循环一个batch，n自增1，最终用acc_sum/n来作为所有batch准确率的均值也就是整个数据集上的准确率
    return acc_sum/n

num_epochs, lr = 5, 0.1
def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y, y_hat).sum()
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)#如果没指定梯度下降算法就用随机梯度下降，如果指定了就用optimizer
            else:
                optimizer.step()
            train_l_sum += l.item()
            train_acc_sum +=(y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch %d, loss %.4f, train acc% .3f, testacc % .3f'% (epoch + 1, train_l_sum / n,train_acc_sum / n, test_acc))
train_ch3(net, train_iter, test_iter,cross_entropy, num_epochs, batch_size, [w, b],lr)
#以上是softmax分类器的训练过程

X, y = iter(test_iter).next()
true_labels =d2l.get_fashion_mnist_labels(y.numpy())#y.numpy()可以将torch中的tensor转化为numpy中的数组
pred_labels =d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
#get_fashion_mnist_labels()返回的是由标签名组成的元组，每个标签名都是一个字符串
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])
#以上是训练结束后的预测（分类）过程

'''小结:
这是实现了一个softmax分类器，其整个训练过程与线性模型是类似的，而且特别地，这个softmax分类器就仅仅是在线性输出上再加了一层softmax layer。
事实上，绝大多数深度学习模型的训练都有着类似的步骤：获取并读取数据、定义模型和损失函数并使用优化算法训练模型。'''

