import time
import torch
from torch import optim,nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import sys
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_images(imgs,num_rows,num_cols,scale=2):#绘图函数
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=d2l.plt.subplots(num_rows,num_cols,figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i*num_cols+j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes

def apply(img,aug,num_rows=2,num_cols=4,scale=1.5):#大多数的图像增广方法具有一定的随机性，为了方便观察图像增广效果，定义该函数apply，该函数对输入图像img做多次增广aug并展示所有结果
    Y=[aug(img) for _ in range(num_rows*num_cols)]#num_rows*num_cols=2*4=8，即做8次增广并将它们均表现出来
    show_images(Y,num_rows,num_cols,scale)

flip_aug=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),torchvision.transforms.ToTensor()])
no_aug=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])#不做任何增广操作，只是转化成tensor

num_workers=0 if sys.platform.startswith('win32') else 4

def load_cifar10(is_train,augs,batch_size,root='D:/新建文件夹/Pycharm Community/datasets/CIFAR-10'):
    dataset=torchvision.datasets.CIFAR10(root=root,train=True,transform=augs,download=False)
    return DataLoader(dataset,batch_size=batch_size,shuffle=is_train,num_workers=num_workers)

def train(train_iter,test_iter,net,loss,optimizer,device,num_epochs):
    print('use',torch.cuda.device_count(),'GPU(s)')
    net=net.cuda()#原文中这里有net=nn.DataParallel(net, device_ids=[0, 1])，当有两张gpu进行并行计算时可以调用，gpu的编号从0开始
    print('training on',device)
    batch_count=0
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n=0.0,0.0,0
        start=time.time()
        for X,y in train_iter:
            X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum+=l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc=d2l.evaluate_accuracy(test_iter,net)
        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.1f sec'%(epoch+1,train_l_sum/batch_count,train_acc_sum/n,test_acc,time.time()-start))

def train_with_data_aug(train_augs,test_augs,lr=0.001):
    batch_size,net=256,d2l.resnet18(10)
    optimizer=optim.Adam(net.parameters(),lr)
    loss=nn.CrossEntropyLoss()
    train_iter=load_cifar10(True,train_augs,batch_size)
    test_iter=load_cifar10(False,test_augs,batch_size)
    train(train_iter,test_iter,net,loss,optimizer,device,num_epochs=10)

train_with_data_aug(flip_aug,no_aug)