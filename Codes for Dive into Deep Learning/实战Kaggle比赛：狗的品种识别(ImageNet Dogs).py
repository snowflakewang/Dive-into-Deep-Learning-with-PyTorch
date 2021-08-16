import time
import torch
from torch import nn,optim
import torchvision
import os
import shutil
import collections
import math
data_dir='D:/新建文件夹/Pycharm Community/datasets/kaggle_dogs/'
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_csv_labels(fname):
    '''读取文件，返回图片名称到标签之间的映射'''
    with open(fname,'r') as f:
        lines=f.readlines()[1:]#跳过表头
        #此处的readlines()函数需要注意，python中有read()，readline()，readlines()三种函数
        #read()
        # 语法：fileObject.read([size])
        # fileObject：打开的文件对象
        # size：可选参数，用于指定一次最多可读取的字符（字节）个数，如果省略，则默认一次性读取所有内容。
        # read()方法用于逐个字节（或者逐个字符）读取文件中的内容，需要借助open() 函数，并以可读模式（包括 r、r+、rb、rb+）打开文件。

        # readline()
        # 语法：fileObject.readline([size])
        # fileObject：打开的文件对象
        # size：可选参数，用于指定读取每一行时，一次最多读取的字节数。
        # readline() 方法用于从文件读取整行，包括 "\n" 字符。readline()读取文件数据的前提是使用open() 函数指定打开文件的模式必须为可读模式（包括 r、rb、r+、rb+）

        # readlines()
        # 语法：fileObject.readlines()
        # fileObject：打开的文件对象
        # readlines() 方法用于一次性读取所有行并返回列表，该列表可以由 Python 的 for... in ... 结构进行处理。
    tokens=[l.rstrip().split(',') for l in lines]
    #python中有三种去除头尾字符，空白符的函数
    #strip： 用来去除头尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #lstrip：用来去除开头字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
    #string.strip([chars])
    #string.lstrip([chars])
    #string.rstrip([chars])
    #参数chars是可选的，当chars为空，默认删除string头尾的空白符(包括\n、\r、\t、' ')
    #当chars不为空时，函数会被chars解成一个个的字符，然后将这些字符去掉。
    #它返回的是去除头尾字符(或空白符)的string副本，string本身不会发生改变。
    return dict(((name,label) for name,label in tokens))

def copyfile(filename,target_dir):
    '''将文件复制到目标路径下'''
    os.makedirs(target_dir,exist_ok=True)#在路径不存在的情况下创建路径，os.makedirs()用来创造多层目录（与此对应地，os.mkdir()只创造单层目录，如果os.mkdir()要创造的目录的之前的根目录有一些是不存在的，就会报错）
    #os.makedirs(name, mode=0o777, exist_ok=False)，name：你想创建的目录名，mode：要为目录设置的权限数字模式，默认的模式为 0o777 (八进制)，
    #exist_ok：是否在目录存在时触发异常。如果exist_ok为False（默认值），则在目标目录已存在的情况下触发FileExistsError异常；如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常
    shutil.copy(filename,target_dir)
    #关于shutil模块，该模块是用来进行文件操作的
    #shutil.copyfile(src,dst)：
    #src是需要被操作的文件名，dst是该文件需要被复制到的地方，注意dst应该是路径加文件名而不是单纯的路径，举个例子：copyfile(data_dir+'xxx.jpg',target_dir+'xxx.jpg')而不能写成copyfile(data_dir+'xxx.jpg',target_dir)
    #shutil.copy(src,dst)：
    #注意这与copyfile()略有不同，src是需要被操作的文件名，dst是文件名或者路径名

def reorg_train_valid(data_dir,labels,valid_ratio):
    n=collections.Counter(labels.values()).most_common()[-1][1]#训练集中样本数最少的类别包含的样本数，
    #其中Counter()统计“可迭代序列”中每个元素的出现的次数，Counter().most_common(n)统计出现次数最多的前n个元素，当n未给定时，列出全部元素及其出现次数
    #本句中，对most_common()数组取最后一个元素（一个二元元组），即得样本数最少的类别，再取元组中的后一个元素，即得到样本数最少的类别包含的样本数
    n_valid_per_label=max(1,math.floor(n*valid_ratio))#验证集中每一类的样本数，其中math.floor(n)函数给出不大于n的最大整数
    #n*valid_ratio代表训练集样本最少的一类可以被作为验证集样本的数量，如果超过了这个数，就会有至少某一类图片在验证集中数量不足的情况
    label_count={}#创建一个字典，其索引用于表征train_valid_test/train_valid/路径下已存在的以label命名的文件夹，其内容用于表征label文件夹中文件数量
    for train_file in os.listdir(os.path.join(data_dir,'train')):
        label=labels[train_file.split('.')[0]]#为每个训练集样本匹配类别，其中.split()通过指定分隔符对字符串进行切片
        #train_file是一个文件名字符串，如xxxx.jpg，因此train_file.split('.')将文件名字符串的后缀与前缀分开，第一元是文件名，第二元是jpg，[0]代表取第一元即文件名，该文件名作为字典labels的索引，给label赋予该文件对应的标签
        fname=os.path.join(data_dir,'train',train_file)#获得train_file文件的路径
        copyfile(fname,os.path.join(data_dir,'train_valid_test','train_valid',label))#将train_file文件保存到train_valid_test/train_valid/路径下的以其label命名的文件夹中
        if label not in label_count or label_count[label]<n_valid_per_label:#如果验证集中还没有这种label或者这种label的验证集样本数还不足
            copyfile(fname,os.path.join(data_dir,'train_valid_test','valid',label))
            label_count[label]=label_count.get(label,0)+1#当label_count字典中能够查询到label索引时，返回label索引对应的值，这对应label_count[label]<n_valid_per_label情况，说明label类别已存在但还未达到所需数量；
            #若不能查询到，返回括号中后面的值，即0，这对应label not in label_count情况，说明label类别还不存在需要先创建，然后label_count.get(label,0)+1=0+1=1意为现在label类别中有一个文件
        else:
            copyfile(fname,os.path.join(data_dir,'train_valid_test','train',label))#构造的验证集达到了类别(label not in label_count)和每个类别中数量(label_count[label]<n_valid_per_label)的要求后，剩下的文件用于构造训练集
    return n_valid_per_label

def reorg_test(data_dir):#用该函数整理测试集，从而方便预测时读取
    for test_file in os.listdir(os.path.join(data_dir,'test')):
        copyfile(os.path.join(data_dir,'test',test_file),os.path.join(data_dir,'train_valid_test','test','unknown'))
        #测试集数据整理，其实就是把data_dir/test中所有文件拷贝到data_dir/train_valid_test/test/unknown中，类别只有一类，名为unknown

def reorg_dog_data(data_dir,valid_ratio):
    labels=read_csv_labels(os.path.join(data_dir,'labels.csv'))#获得字典labels，索引是图片文件名称，键值是类别
    reorg_train_valid(data_dir,labels,valid_ratio)#将原训练集train（或者说train_valid，因为对原训练集train分类后得到train_valid）分割成新的训练集train和验证集valid并与标签集labels匹配
    reorg_test(data_dir)#整理出测试集test，里面包含仅一类成为unknown

batch_size,valid_ratio=128,0.1
reorg_dog_data(data_dir,valid_ratio)

#以下是图像增广
transform_train=torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(224,scale=(0.08,1.0),ratio=(3.0/4.0,4.0/3.0)),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.4),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
transform_test=torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                               torchvision.transforms.CenterCrop(224),
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])

#以下是将原始图片文件经过图像增广和张量转化的过程，转化成可以输入网络的变量
train_ds,train_valid_ds=[torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_train) for folder in ['train','train_valid']]
valid_ds,test_ds=[torchvision.datasets.ImageFolder(os.path.join(data_dir,'train_valid_test',folder),transform=transform_test) for folder in ['valid','test']]
#关于这里使用的ImageFolder，它是一个通用的数据加载器，它要求以data_dir/xxx.jpg（或者是png等其他格式）来组织数据集的训练、验证和测试
#先来看看torchvision.datasets.ImageFolder(root,transform,target_transform,loader)，root代表图片存储的根目录，即各类别文件夹所在目录的上一层目录；transform是对图片进行的预处理操作（函数），
#原始图片作为输入，返回转换后的图片；target_transform是对图片类别进行的预处理操作，输入为 target，输出对其的转换。如果不传该参数，即对 target 不做任何转换，返回的顺序索引 0,1, 2…
#该函数的返回值有三种属性，
#为self.classes：用一个 list 保存类别名称
#self.class_to_idx：类别对应的索引，与不做任何转换返回的 target 对应
#self.imgs：保存(img_path, class) tuple的 list，其中img_path指的是文件的路径

train_iter,train_valid_iter=[torch.utils.data.DataLoader(dataset,batch_size,shuffle=True,drop_last=True) for dataset in (train_ds,train_valid_ds)]
test_iter=torch.utils.data.DataLoader(test_ds,batch_size,shuffle=False,drop_last=False)
valid_iter=torch.utils.data.DataLoader(valid_ds,batch_size,shuffle=False,drop_last=True)

#定义网络结构，使用在ImageNet上预训练过的ResNet-34
def get_net(devices):
    finetune_net=nn.Sequential()
    finetune_net.features=torchvision.models.resnet34(pretrained=True)
    finetune_net.output_new=nn.Sequential(nn.Linear(1000,256),nn.ReLU(),nn.Linear(256,120))
    finetune_net=finetune_net.to(devices)
    #print(finetune_net)直接打印finetune_net会得到features和output_new两部分，这两部分都是由网络层构成的类
    #print(finetune_net.features)打印finetune_net.features只会得到features部分
    for params in finetune_net.features.parameters():
        params.requires_grad=False
    return finetune_net

#使用交叉熵损失函数
loss=nn.CrossEntropyLoss()#用的公式是H(p,q)=-sigma(p*log(q)),q=softmax(x)

#计算损失的函数
def evaluate_loss(data_iter,net,devices):
    l_sum,n=0.0,0
    for features,labels in data_iter:
        features,labels=features.to(devices),labels.to(devices)
        output=net(features)
        l=loss(output,labels)
        l_sum=l.sum()
        n+=labels.numel()
    return  l_sum/n

#计算准确率的函数
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

#训练网络的函数
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            #y.backward()函数可以用来计算scalar对tensor的导数，而且只能用来计算这个，如果y是一个tensor而且括号里什么都不写，那么就会报错
            #这是因为tensor对tensor的导数计算是不明确的比较复杂，因此在括号里写一个与y同形的tensor就是要转化成scalar对tensor求导的形式
            #举个例子，如果y=[y1,y2,y3]，则要传入一个和y同形状的tensor，设为w=[w1,w2,w3]
            #那么y.backward(w)计算的并不是y对x的导数，而是l=torch.sum(y*w)=y1*w1+y2*w2+y3*w3对x的导数
            #此处的w有点类似于权重的感觉，其目的就是把tensor转化为scalar，这样就把dy/dx这种tensor对tensor的求导转化为了dy1/dx,dy2/dx,dy3/dx
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))

#超参数、网络结构、优化器、训练函数的定义
devices,num_epochs,lr,wd=device,5,0.01,1e-4
net=get_net(devices)
optimizer=optim.SGD(net.parameters(),lr,momentum=0.9,weight_decay=wd)
#注意这里使用了weight_decay，意为权重衰减，在SGD中可以理解为通过引入L2范数做到的，L2范数为(lambda/2)*sigma(i) w_i**2，
#损失函数改写为loss:=loss+L2_Norm，则grad(loss)/grad(w_i)=g(loss)/g(w_i)+lambda*w_i，权重更新公式为
#w_i:=w_i-(g(loss)/g(w_i)+lambda*w_i)=w_i*(1-lambda)-g(loss)/g(w_i)，1-lambda这样一个小于1的数相当于实现了权重衰减
#这里的参数weight_decay就是前面提到的lambda

#需要注意的是，L2正则化本身并不等于权重衰减，只是在标准SGD算法中它们恰好是一样的。但是例如在Adam算法中它们就是不一样的，这个可以参考文章
#Decoupled Weight Decay Regularization
train(train_iter,test_iter,net,loss,optimizer,device,num_epochs)