from cmath import sqrt
import time
from tkinter import N
from sympy import mod_inverse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeEmbeddings(nn.Module):
    '''
    Time embedding module for diffusion models.
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim=dim

    def forward(self, time):
        half_dim = int(self.dim // 2)
        freq = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(0, half_dim, 1, device=time.device) * -freq)
        emb = time[:, None] * freqs[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb
class ResidualBlock(nn.Module):
    '''残差链接层'''
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
        # conv2 should map out_channels -> out_channels
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.time_mlp=nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim,out_channels)
        )
        self.norm1=nn.GroupNorm(8,in_channels)
        self.norm2=nn.GroupNorm(8,out_channels)

        if in_channels!=out_channels:
            self.residual=nn.Conv2d(in_channels,out_channels,1)
        else:
            self.residual=nn.Identity()
    
    def forward(self,x,time_emb):
        #normalize->activate->convolve
        h=self.conv1(F.silu(self.norm1(x)))
        h=h+self.time_mlp(time_emb)[:,:,None,None]
        h=self.conv2(F.silu(self.norm2(h)))
        return h+self.residual(x)
class AttentionBlock(nn.Module):
    '''自注意力层'''
    def __init__(self, channel):
        super().__init__()
        self.norm=nn.GroupNorm(8,channel)
        self.q=nn.Conv2d(channel,channel,1)
        self.k=nn.Conv2d(channel,channel,1)
        self.v=nn.Conv2d(channel,channel,1)
        self.out=nn.Conv2d(channel,channel,1)

    def forward(self, x):
        batch, channels, height, width = x.shape
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1)  #便于q@k,批量矩阵乘法，permute交换维度
        k = k.reshape(batch, channels, height * width)
        v = v.reshape(batch, channels, height * width).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (channels ** (-0.5))  #将方差控制到1，避免梯度消失
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(batch, channels, height, width)
        # use defined output conv
        h = self.out(h)

        return x + h
class UNet(nn.Module):
    '''Unet架构,适用于28*28,channe=64'''
    def __init__(self, in_channel=1, model_channel=64, num_res_block=2):  #MNIST是单通道灰度图->in_channel=1 else:in_channel=3
        super().__init__()
        self.in_channel=in_channel
        self.model_channel=model_channel
        time_emb_dim=model_channel*4  #经验性设计，足够多的维度表示时间
        self.time_emb=nn.Sequential(  #时间嵌入层，通过一个MLP更适配
            TimeEmbeddings(model_channel),
            nn.Linear(model_channel,time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim,time_emb_dim)
        )
        #init_conv;
        self.init_conv=nn.Conv2d(in_channel,model_channel,kernel_size=3,stride=1,padding=1)
        #编码器：28*28->14*14->7*7  64->128->256
        #level 0:28*28  64channels
        self.level0_blocks = nn.ModuleList(
            ResidualBlock(model_channel, model_channel, time_emb_dim) for _ in range(num_res_block)
        )
        self.level0_down = nn.Conv2d(model_channel, model_channel*2, kernel_size=3, stride=2, padding=1)
        #level 1:14*14  128channels
        self.level1_blocks = nn.ModuleList(
            ResidualBlock(model_channel*2, model_channel*2, time_emb_dim) for _ in range(num_res_block)
        )
        self.level1_down = nn.Conv2d(model_channel*2, model_channel*4, kernel_size=3, stride=2, padding=1)
        #level 2:7*7  256channels
        self.level2 = nn.ModuleList(
            ResidualBlock(model_channel*4, model_channel*4, time_emb_dim) for _ in range(num_res_block)
        )

        #中间层
        self.attn=AttentionBlock(model_channel*4)
        self.resi=ResidualBlock(model_channel*4,model_channel*4,time_emb_dim)

        #解码器
        #level 3:7*7->14*14 256channels
        self.level3_blocks = nn.ModuleList([
            ResidualBlock(model_channel*4+model_channel*4, model_channel*4, time_emb_dim),
            ResidualBlock(model_channel*4, model_channel*4, time_emb_dim)
        ])
        self.level3_up = nn.ConvTranspose2d(model_channel*4, model_channel*2, kernel_size=4, stride=2, padding=1)
        #level 4:14*14->28*28  128channels
        self.level4_blocks = nn.ModuleList([
            ResidualBlock(model_channel*2+model_channel*2, model_channel*2, time_emb_dim),
            ResidualBlock(model_channel*2, model_channel*2, time_emb_dim)
        ])
        self.level4_up = nn.ConvTranspose2d(model_channel*2, model_channel, kernel_size=4, stride=2, padding=1)
        #level 5:28*28  64channels
        self.level5_block1 = ResidualBlock(model_channel+model_channel, model_channel, time_emb_dim)
        self.level5_block2 = ResidualBlock(model_channel, model_channel, time_emb_dim)
        
        self.out_block = ResidualBlock(model_channel, model_channel, time_emb_dim)
        self.out_norm = nn.GroupNorm(8, model_channel)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(model_channel, in_channel, kernel_size=3, padding=1)

    def forward(self, x, timesteps):
        time_emb=self.time_emb(timesteps)
        #init conv
        h=self.init_conv(x)
        #down sample
        #level 0->level 1
        skip_0=h
        for res_block in self.level0_blocks:
            h=res_block(h,time_emb)
        h = self.level0_down(h)

        #level1->level2
        skip_1=h
        for res_block in self.level1_blocks:
            h=res_block(h,time_emb)
        h = self.level1_down(h)

        #level2->level3
        skip_2=h
        for res_block in self.level2:
            h=res_block(h,time_emb)
        h = self.attn(h)
        h = self.resi(h,time_emb)

        #up sample
        #level3->level4
        h = torch.cat([h, skip_2], dim=1) #跳跃连接
        for res_block in self.level3_blocks:
            h=res_block(h,time_emb)
        h = self.level3_up(h)
        
        #level4->level5
        h = torch.cat([h, skip_1], dim=1)
        for res_block in self.level4_blocks:
            h=res_block(h,time_emb)
        h = self.level4_up(h)
        
        #level5
        h = torch.cat([h, skip_0], dim=1)
        h = self.level5_block1(h, time_emb)
        h = self.level5_block2(h, time_emb)
        #output
        h = self.out_block(h, time_emb)
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h
class DDPMNet(nn.Module):
    def __init__(self,model, beta_start=1e-4, beta_end=2e-2, timesteps=1000, device='cpu' ):
        '''没有可训练的参数  参数在unet中  创建的tensor都要to(device)'''
        super().__init__()
        self.model=model
        self.device=device
        self.timesteps=timesteps
        #计算beta,alpha等系数
        self.beta=torch.linspace(beta_start,beta_end,timesteps).to(device)
        self.alpha=1.0-self.beta
        self.alpha_cumprod=torch.cumprod(self.alpha,dim=0)
        self.alpha_cumprod_prev=torch.cat([torch.tensor([1.0]).to(device),self.alpha_cumprod[:-1]],dim=0)
       # 计算用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        # 计算用于去噪的系数
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alpha)
        self.posterior_variance = self.beta * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        '''前向加噪过程'''
        if noise is None:
            noise = torch.randn_like(x_start).to(self.device)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t):
        '''反向去噪过程'''
        with torch.no_grad():
            predicted_noise = self.model(x_t, t)
            coef1 = 1 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1)
            coef2 = (self.beta[t] / torch.sqrt(1 - self.alpha_cumprod[t])).view(-1, 1, 1, 1)
            mean = coef1 * (x_t - coef2 * predicted_noise)
            if t[0] == 0:
                return mean
            else:
                noise = torch.randn_like(x_t).to(self.device)
                variance = self.posterior_variance[t].view(-1, 1, 1, 1)
                return mean + torch.sqrt(variance) * noise
    
    def sample(self,batch_size,img_size=28):
        '''从纯噪声生成图像'''
        self.model.eval()
        img = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long).to(self.device)  #每一个时间步都对一个batch采样
            img = self.p_sample(img, t)
        return img
    
    def compute_loss(self, x_start):
        '''计算损失函数'''
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()  #对batch里每个样本都抽取一个t
        noise = torch.randn_like(x_start).to(self.device)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy, t)
        return F.mse_loss(noise, predicted_noise)