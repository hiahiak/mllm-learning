from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.norm1=nn.GroupNorm(2,in_channel)
        self.act1=nn.SiLU()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,padding=1)
        self.norm2=nn.GroupNorm(1,out_channel)
        self.act2=nn.SiLU()
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        self.shortcut=nn.Conv2d(in_channel,out_channel,kernel_size=1) if in_channel!=out_channel else nn.Identity()

    def forward(self,x):
        h=self.conv1(self.act1(self.norm1(x)))
        h=self.conv2(self.act2(self.norm2(h)))
        return h+self.shortcut(x)

class Encoder(nn.Module):
    '''[3,256,256]->[6,32,32]'''
    def __init__(self, in_channel=3,latent_dim=32):
        super().__init__()
        self.out_channel=4
        self.conv0=nn.Conv2d(in_channel,latent_dim,kernel_size=3,padding=1)
        self.res1=ResidualBlock(latent_dim,latent_dim)
        self.conv1=nn.Conv2d(latent_dim,latent_dim,kernel_size=3,stride=2,padding=1)
        self.res2=ResidualBlock(latent_dim,latent_dim*2)
        self.conv2=nn.Conv2d(latent_dim*2,latent_dim*2,kernel_size=3,stride=2,padding=1)
        self.res3=ResidualBlock(latent_dim*2,latent_dim*4)
        self.conv3=nn.Conv2d(latent_dim*4,latent_dim*4,kernel_size=3,stride=2,padding=1)
        self.res4=ResidualBlock(latent_dim*4,latent_dim*4)
        self.conv_mu=nn.Conv2d(latent_dim*4,self.out_channel,kernel_size=3,padding=1)
        self.conv_logvar=nn.Conv2d(latent_dim*4,self.out_channel,kernel_size=3,padding=1)
    def forward(self,x):
        h=self.conv0(x)
        h=self.res1(h)
        h=self.conv1(h)
        h=self.res2(h)
        h=self.conv2(h)
        h=self.res3(h)
        h=self.conv3(h)
        h=self.res4(h)
        mu=self.conv_mu(h)
        logvar=self.conv_logvar(h)
        return mu,logvar
    
class Decoder(nn.Module):
    '''4,32,32]->[3,256,256]'''
    def __init__(self, in_channel=4,latent_dim=32,out_channel=3):
        super().__init__()
        self.initconv=nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1)
        self.res1=ResidualBlock(in_channel,latent_dim*4)
        self.conv1=nn.Conv2d(latent_dim*4,latent_dim*4,kernel_size=3,padding=1)
        self.res2=ResidualBlock(latent_dim*4,latent_dim*2)
        self.conv2=nn.Conv2d(latent_dim*2,latent_dim*2,kernel_size=3,padding=1)
        self.res3=ResidualBlock(latent_dim*2,latent_dim)
        self.conv3=nn.Conv2d(latent_dim,latent_dim,kernel_size=3,padding=1)
        self.res4=ResidualBlock(latent_dim,out_channel)
        self.outconv=nn.Conv2d(out_channel,out_channel,kernel_size=3,padding=1)
        self.act=nn.Sigmoid() #与dataset中初始x的数据范围对齐[0,1]->才能计算mse
    
    def forward(self,x):
        x=self.initconv(x)
        x=self.res1(x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.conv1(x)
        x=self.res2(x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.conv2(x)
        x=self.res3(x)
        x=F.interpolate(x,scale_factor=2,mode='bilinear')
        x=self.conv3(x)
        x=self.res4(x)
        x=self.outconv(x)
        x=self.act(x)
        return x
    
class VAE(nn.Module):
    def __init__(self, in_channel=3,out_channel=4,latent_dim=32):
        super().__init__()
        self.encoder=Encoder(in_channel,latent_dim)
        self.decoder=Decoder(out_channel,latent_dim,in_channel)

    def compute_loss(self,x):
        mu,logvar=self.encoder(x)
        std=torch.exp(0.5*logvar)
        eps=torch.rand_like(std)
        z=mu+std*eps
        x_decoder=self.decoder(z)
        loss1=F.mse_loss(x,x_decoder)
        loss2=-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss1+loss2
     
    def forward(self,x):
        mu,logvar=self.encoder(x)
        std=torch.exp(0.5*logvar)
        eps=torch.rand_like(std)
        z=mu+std*eps
        x_decoder=self.decoder(z)
        loss=self.compute_loss(x)
        return x_decoder,z
    
    
    