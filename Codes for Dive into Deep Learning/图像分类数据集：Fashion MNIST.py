'''this file is related to 图像分类数据集：Fashion MNIST in PyTorch 深度学习：60分钟快速入门（官方）'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import sys
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
import d2lzh_pytorch as d2l

mnist_train=torchvision.datasets.FashionMNIST(root='D:/新建文件夹/Pycharm Community/pythonProject/datasets',train=True,download=False,transform=transforms.ToTensor())
mnist_test=torchvision.datasets.FashionMNIST(root='D:/新建文件夹/Pycharm Community/pythonProject/datasets',train=False,download=False,transform=transforms.ToTensor())
#将路径D:/新建文件夹/Pycharm Community/pythonProject/datasets/FashionMNIST/processed下的文件train.py改名为training.py，就可以加载本地数据
print(type(mnist_train))
print(len(mnist_train),len(mnist_test))

def get_fashion_mnist_labels(labels):
    text_labels= ['t-shirt', 'trouser',
'pullover', 'dress', 'coat',
 'sandal', 'shirt',
'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

batch_size=256
if sys.platform.startswith('win'):
    num_workers=0#0表示不用额外进程读取数据
else:
    num_workers=4
train_iter=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
train_test=torch.utils.data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)



