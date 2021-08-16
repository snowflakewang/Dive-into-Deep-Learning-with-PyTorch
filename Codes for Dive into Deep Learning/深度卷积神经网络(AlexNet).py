import torch
from torch import nn,optim
import d2lzh_pytorch.utils as d2l
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,96,11,4),nn.ReLU(),nn.MaxPool2d(3,2),nn.Conv2d(96,256,5,1,2),nn.ReLU(),nn.MaxPool2d(3,2),
                                nn.Conv2d(256,384,3,1,1),nn.ReLU(),nn.Conv2d(384,384,3,1,1),nn.ReLU(),nn.Conv2d(384,256,3,1,1),nn.ReLU(),nn.MaxPool2d(3,2))
        self.fc=nn.Sequential(nn.Linear(256*5*5,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),nn.Linear(4096,10))

    def forward(self,img):
        feature=self.conv(img)
        output=self.fc(feature.view(img.shape[0],-1))
        return output

net=AlexNet()
batch_size=128
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size,resize=224)

lr,num_epochs=0.001,5
optimizer=optim.Adam(net.parameters(),lr)
d2l.train_ch5(net,train_iter,test_iter,batch_size,optimizer,device,num_epochs)