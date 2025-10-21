from cProfile import label
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  #导入进度条工具，用于显示训练、数据处理进度

def train_epoch(ddpm,dataloader,optimizer,device,epoch,vae):
    ddpm.model.train()
    running_loss=0.0
    num_batches=len(dataloader)
    loop=tqdm(dataloader,desc=f'Epoch:{epoch+1}',leave=False)  #description，leave=false训练结束后不保留进度条显示

    for batch_idx,(data,label) in enumerate(loop):
        data=data.to(device)
        label=label.to(device)
        data=data*2.0-1.0  #数据处理时候ToTensor将图片归一化到(0,1)，先要缩放到（-1.0，1.0）让图像在加噪前就接近零均值对称
        _,data=vae(data)
        optimizer.zero_grad()
        batch_loss=ddpm.compute_loss(data,label)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(),1.0)  #梯度裁剪，>1.0按比例缩放
        optimizer.step()
        running_loss+=batch_loss.item()

        loop.set_postfix({
            'ddpm_loss':f'{batch_loss.item():.6f}',
            'ddpm_avg_loss':f'{running_loss/(batch_idx+1):.6f}'
        })

    return running_loss/num_batches

def vae_train_epoch(vae,dataloader,optimizer,device,epoch):
    vae.train()
    running_loss=0.0
    num_batches=len(dataloader)
    loop=tqdm(dataloader,desc=f'Epoch:{epoch+1}',leave=False)
    for batch_idx,(data,_) in enumerate(loop):
        data=data.to(device)
        data=data*2-1.0
        optimizer.zero_grad()
        batch_loss=vae.compute_loss(data)
        batch_loss.backward()
        nn.utils.clip_grad_norm_(vae.parameters(),1.0) 
        optimizer.step()
        running_loss+=batch_loss.item()
        loop.set_postfix({
            'vae_loss':f'{batch_loss.item():.6f}',
            'vae_avg_loss':f'{running_loss/(batch_idx+1):.6f}'
        })
    return running_loss/num_batches
