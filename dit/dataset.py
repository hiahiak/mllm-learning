import os
import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split  
from PIL import Image

def get_loaders():
    print("preparing data loaders...")
    #define transform for train&test sets
    transform=transforms.Compose([  #Compose([])将一系列预处理操作按顺序打包
        transforms.Resize(256),
        transforms.CenterCrop(256),  
        transforms.ToTensor(), #转化为tensor，并且将像素值缩放到(0.0,1.0)
    ])
    #在当前文件的目录的上一级目录创建data文件夹
    data_root=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','data'))
    os.makedirs(data_root,exist_ok=True)

    full_train_set=datasets.CIFAR10(root='../data',train=True,transform=transform,download=True) 
    test_set=datasets.CIFAR10(root='../data',train=False,transform=transform,download=True)

    train_size=int(0.8*len(full_train_set))
    val_size=len(full_train_set)-train_size
    train_set,val_set=random_split(full_train_set,[train_size,val_size])

    train_loader=DataLoader(train_set,batch_size=8,shuffle=True,num_workers=2) #num_workers=2表示使用两个子进程来并行加载数据
    val_loader=DataLoader(val_set,batch_size=8,shuffle=False,num_workers=2)
    test_loader=DataLoader(test_set,batch_size=8,shuffle=False,num_workers=2)

    return train_loader,val_loader,test_loader