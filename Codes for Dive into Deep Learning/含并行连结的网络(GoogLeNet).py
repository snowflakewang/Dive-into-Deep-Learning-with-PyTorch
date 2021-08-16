import torch
from torch import nn, optim
import torch.nn.functional as f
import d2lzh_pytorch.utils as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Inception(nn.Module):
    # c1到c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)  # 线路1是单1*1卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)  # 线路2是1*1卷积层后接3*3卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)  # 线路3是1*1卷积层后接3*3卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = f.relu(self.p1_1(x))# 这里还可以写成p1=(self.p1_1(x)).relu()。原文中的f.relu()单纯就是一个函数，它在源文件中是用def来定义的
        # 下面来看两个相近的模块，一个是torch.nn，一个是torch.nn.functional
        # torch.nn和torch.nn.functional里面有很多是对应的，比如说torch.nn.ReLU()和torch.nn.functional.relu()，它们都在某种程度上代表激活函数(activation function)Relu函数
        # torch.nn.Conv2d和torch.nn.functional.conv2d也是类似的道理，但是要注意，前者是指的是一个二维卷积层，是一个类；而后者指的是单纯一个函数
        # 而且nn.Conv2d这个类中的forward函数里会调用nn.functional.conv2d函数来完成前向传播计算
        # 注意：如果模型有可学习的参数时，最好使用nn.Module；否则既可以使用nn.functional也可以使用nn.Module，二者在性能上没有太大差异，具体的使用方式取决于个人喜好，
        # 由于激活函数（ReLu、sigmoid、Tanh）、池化（MaxPool）等层没有可学习的参数，可以使用对应的functional函数，而卷积、全连接等有可学习参数的网络建议使用nn.Module
        # 虽然dropout没有可学习参数，但建议还是使用nn.Dropout而不是nn.functional.dropout，因为dropout在训练和测试两个阶段的行为有所差别，使用nn.Module对象能够通过model.eval操作加以区分

        # 在代码中，不具备可学习参数的层（激活层、池化层），将它们用函数代替，这样可以不用放置在构造函数__init__中
        # 有可学习的模块，也可以用functional代替，只不过实现起来比较繁琐，需要手动定义参数parameter，
        # 如下面实现自定义的全连接层，就可以将weight和bias两个参数单独拿出来，在构造函数中初始化为parameter
        ''' class mylinear(nn.Module):
                def __init__(self):
                    super(mylinear, self).__init__()
                    self.weight=nn.Parameter(torch.randn((3,4)))
                    self.bias=nn.Parameter(torch.zeros(3))

                def forward(self,input):
                    return f.linear(input,self.weight,self.bias)'''

        p2 = f.relu(self.p2_2(f.relu(self.p2_1(x))))
        p3 = f.relu(self.p3_2(f.relu(self.p3_1(x))))
        p4 = f.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)


blk1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 192, kernel_size=3, padding=1),
                     nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
blk3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32), Inception(256, 128, (128, 192), (32,96),64),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
blk4=nn.Sequential(Inception(480,192,(96,208),(16,48),64),Inception(512,160,(112,224),(24,64),64),Inception(512,128,(128,256),(24,64),64),
                   Inception(512,112,(144,288),(32,64),64),Inception(528,256,(160,320),(32,128),128),nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
blk5=nn.Sequential(Inception(832,256,(160,320),(32,128),128),Inception(832,384,(192,384),(48,128),128),d2l.GlobalAvgPool2d())
net=nn.Sequential(blk1,blk2,blk3,blk4,blk5,d2l.FlattenLayer(),nn.Linear(1024,10))

batch_size=128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=96)
num_epochs,lr=5,0.001
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)