import torch
from torch import optim,nn
from torch.utils.data import Dataset,DataLoader
import torchvision
from PIL import Image
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img=Image.open('D:/我的图像/AI英才班.png')
d2l.plt.imshow(img)

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

apply(img,torchvision.transforms.RandomHorizontalFlip())#水平翻转

apply(img,torchvision.transforms.RandomVerticalFlip())#垂直翻转

shape_aug=torchvision.transforms.RandomResizedCrop(200,scale=(0.1,1),ratio=(0.5,2))#随机裁剪出原面积10%-100%的区域，且该区域H:W在0.5-2之间，然后将H和W分别缩放到200pixel
apply(img,shape_aug)

color_aug=torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.5,saturation=0.5,hue=0.5)#brightness亮度，contrast对比度，saturation饱和度，hue色调
apply(img,color_aug)

augs=torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(),color_aug,shape_aug])#可以通过aug将多个图像增广方法叠加使用
apply(img,augs)