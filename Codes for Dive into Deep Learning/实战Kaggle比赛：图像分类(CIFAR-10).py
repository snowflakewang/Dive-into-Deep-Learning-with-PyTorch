import collections
import math
import torch
from torch import nn,optim
import torchvision
import os
import shutil
import d2lzh_pytorch.utils as d2l
data_dir='D:/新建文件夹/Pycharm Community/动手学深度学习/CIFAR-10'

def read_csv_labels(fname):
    '''读取文件，返回图片名称到标签之间的映射'''
    with open(fname,'r') as f:
        lines=f.readlines()[1:]#跳过表头
    tokens=[l.rstrip().split(',') for l in lines]
    return dict(((name,label) for name,label in tokens))

'''labels=read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))#os.path.join(path,*paths)意为组合path与paths,返回一个路径字符串
print('training examples:',len(labels))#显示训练样本数
print('classes:',len(set(labels.values())))#显示可分出的类别数'''

def copyfile(filename,target_dir):
    '''将文件复制到目标路径下'''
    os.makedirs(target_dir,exist_ok=True)#在路径不存在的情况下创建路径，os.makedirs()用来创造多层目录（与此对应地，os.mkdir()只创造单层目录，如果os.mkdir()要创造的目录的之前的根目录有一些是不存在的，就会报错）
    #os.makedirs(name, mode=0o777, exist_ok=False)，name：你想创建的目录名，mode：要为目录设置的权限数字模式，默认的模式为 0o777 (八进制)，
    #exist_ok：是否在目录存在时触发异常。如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常
    shutil.copy(filename,target_dir)

def reorg_train_valid(data_dir,labels,valid_ratio):
    n=collections.Counter(labels.values()).most_common()[-1][1]#训练集中样本数最少的类别包含的样本数，
    #其中Counter()统计“可迭代序列”中每个元素的出现的次数，Counter().most_common(n)统计出现次数最多的前n个元素，当n未给定时，列出全部元素及其出现次数
    #本句中，对most_common()数组取最后一个元素（一个二元元组），即得样本数最少的类别，再取元组中的后一个元素，即得到样本数最少的类别包含的样本数
    n_valid_per_label=max(1,math.floor(n*valid_ratio))#验证集中每一类的样本数，其中math.floor(n)函数给出不大于n的最大整数
    #n*valid_ratio代表训练集样本最少的一类可以被作为验证集样本的数量，如果超过了这个数，就会有至少某一类图片在验证集中数量不足的情况
    label_count={}
    for train_file in os.listdir(os.path.join(data_dir,'train','train')):
        label=labels[train_file.split('.')[0]]#为每个训练集样本匹配类别，其中.split()通过指定分隔符对字符串进行切片
        #train_file.split('.')是一个二元元组，第一元是文件名，第二元是文件路径，[0]代表取第一元即文件名，该文件名作为字典labels的索引，给label赋予该文件对应的标签
        fname=os.path.join(data_dir,'train','train',train_file)#以类别名为文件夹名称来保存数据
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))
        if label not in label_count or label_count[label]<n_valid_per_label:#如果验证集中还没有这种label或者这种label的验证集样本数还不足
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))
            label_count[label]=label_count.get(label,0)+1
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))
    return n_valid_per_label

def reorg_test(data_dir):#用该函数整理测试集，从而方便预测时读取
    for test_file in os.listdir(os.path.join(data_dir,'test','test')):
        copyfile(os.path.join(data_dir,'test','test',test_file),os.path.join(data_dir,'train_valid_test','test','unknown'))#测试集数据整理，类别名为unknown

def reorg_cifar10_data(data_dir,valid_ratio):#该函数用于调用前面诸读取数据函数
    labels=read_csv_labels(os.path.join(data_dir,'trainLabels.csv'))
    reorg_train_valid(data_dir,labels,valid_ratio)
    reorg_test(data_dir)

batch_size,valid_ratio=128,0.1#批量大小为128，10%的训练样本作为调参用的验证集
reorg_cifar10_data(data_dir,valid_ratio)

transform_train=torchvision.transforms.Compose([torchvision.transforms.Resize(40),#将图像放大成40*40pixel的正方形
                                                torchvision.transforms.RandomResizedCrop(32,scale=(0.64,1.0),ratio=(1.0,1.0)),#随机裁剪出面积为原面积0.64-1倍的H:W=1:1即小正方形，再缩放为32*32pixel的正方形
                                                torchvision.transforms.RandomHorizontalFlip(),#随机进行水平翻转
                                                torchvision.transforms.ToTensor(),#转化为tensor形式可以被网络接受
                                                torchvision.transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])])#对每个pixel的RGB值做归一化
transform_test=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.4914,0.4822,0.4465],std=[0.2023,0.1994,0.2010])])

#通过创建ImageFolder实例来读取整理后的含原始图像文件的数据集，其中每个数据样本包括图像和标签
train_ds,train_valid_ds=[torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_train) for folder in ['train','train_valid']]
valid_ds,test_ds=[torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_test) for folder in ['valid','test']]

train_iter,train_valid_iter=[torch.utils.data.DataLoader(dataset,batch_size,shuffle=True,drop_last=True) for dataset in (train_ds,train_valid_ds)]
valid_iter=torch.utils.data.DataLoader(valid_ds,batch_size,shuffle=False,drop_last=True)
test_iter=torch.utils.data.DataLoader(valid_ds,batch_size,shuffle=False,drop_last=False)
#由于一个数据集的大小不一定刚好是batch_size的整数倍，所以按照batch_size进行分割后，最后一个batch的大小不一定是batch_size，而drop_last的含义是是否要丢弃这最后一个batch。当选择True时，就丢弃；当False时，就不丢弃

def get_net():#定义模型，这里使用ResNet-18网络
    num_classes=10
    net=d2l.resnet18(num_classes,in_channels=3)#RGB是三通道，所以输入通道数是3
    return net

net=get_net()
loss=nn.CrossEntropyLoss()

device,num_epochs,lr,wd='cuda',5,0.1,5e-4
optimizer=optim.SGD(net.parameters(),lr,momentum=0.9,weight_decay=wd)
d2l.train(test_iter,valid_iter,net,loss,optimizer,device,num_epochs)
#CIFAR-10原始的图片标签数据集较大，并不方便训练，因此未做实际训练尝试

