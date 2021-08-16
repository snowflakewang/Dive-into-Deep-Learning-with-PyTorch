import torch
from torch import nn,optim
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def vgg_block(num_convs,in_channels,out_channels):
    blk=[]
    for i in range(num_convs):
        if i==0:
            blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))#因为第一层的输入来自于之前的层，所以其in_channels参数比较特殊，要写一个if else语句来单独定义第一个卷积层
        else:
            blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*blk)

def fc_block(fc_features,fc_hidden_units):
    fc=[]
    fc.append(nn.Flatten(start_dim=1,end_dim=-1))
    #fc.append(d2l.FlattenLayer())，start_dim和end_dim是该张量需要展平的维度，每一个batch都是一个四维的tensor，第一维是batch的大小，后面三维分别是通道数、高、宽，
    #因此需要展平的是后三维，也就是把每一张图片生成的所有feature map展平并连接到一起
    fc.append(nn.Linear(fc_features,fc_hidden_units))
    fc.append(nn.ReLU())
    fc.append(nn.Dropout(0.5))
    fc.append(nn.Linear(fc_hidden_units,fc_hidden_units))
    fc.append(nn.ReLU())
    fc.append(nn.Dropout(0.5))
    fc.append(nn.Linear(fc_hidden_units,10))
    return nn.Sequential(*fc)

def vgg_net(conv_archs,fc_features,fc_hidden_units):
    net=nn.Sequential()
    for i,(num_convs,in_channels,out_channels) in enumerate(conv_archs):
        net.add_module('vgg_block'+str(i+1),vgg_block(num_convs,in_channels,out_channels))
    net.add_module('fc',fc_block(fc_features,fc_hidden_units))
    return net

conv_archs=((1,1,64),(1,64,128),(2,128,256),(2,256,512),(2,512,512))
fc_features=512*7*7# 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_hidden_units=4096

'''因为VGG-11计算上比AlexNet更加复杂，出于测试的目的我们构造一个通道数更小，或者说更窄的网络在Fashion-MNIST数据集上进行训练'''
ratio = 8
small_conv_archs = [(1, 1, 64//ratio), (1,64//ratio, 128//ratio), (2, 128//ratio,256//ratio), (2, 256//ratio, 512//ratio),(2, 512//ratio, 512//ratio)]
net = vgg_net(small_conv_archs, fc_features //ratio, fc_hidden_units // ratio)

batch_size=64
num_epochs,lr=5,0.001
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)