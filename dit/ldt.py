from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from dit import DiT

class Patchfyemb(nn.Module):
    def __init__(self, channel=4,img_size=32,patch_size=2,hidden_cha=768):
        super().__init__()
        self.patch_size=patch_size
        self.proj=nn.Linear(channel*patch_size*patch_size,hidden_cha)
        self.pos=nn.Parameter(torch.zeros(1,int((img_size/patch_size))**2,hidden_cha))
    def forward(self,x):
        B,C,H,W=x.shape
        h=H//self.patch_size
        w=W//self.patch_size
        x=x.view(B,C,h,self.patch_size,w,self.patch_size)
        x=x.permute(0,2,4,1,3,5).contiguous()
        x=x.view(B,h*w,C*self.patch_size*self.patch_size)
        x=self.proj(x) #[b,num_patch,hid_cha]
        return x+self.pos
    
class TimeEmbeddings(nn.Module):
    def __init__(self,emb_dim,device):
        super().__init__()
        self.dim=emb_dim
        self.device=device
        self.mlp=nn.Sequential(
            nn.Linear(emb_dim,emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim,emb_dim)
        )
    def forward(self,t):
        half_dim=self.dim//2
        emb=(torch.log(torch.tensor(10000.0)) / (half_dim - 1)).to(self.device)
        emb=torch.exp(torch.arange(half_dim,dtype=torch.float32,device=self.device)*-emb)
        emb=t[:,None]*emb[None,:]
        emb=torch.cat((emb.sin(),emb.cos()),dim=-1)
        emb=self.mlp(emb)
        return emb

class Unpatchfy(nn.Module):
    def __init__(self, channel=4,img_size=32,patch_size=2,hidden_cha=768):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.channel=channel
        self.norm=nn.LayerNorm(hidden_cha)
        self.proj=nn.Linear(hidden_cha,channel*patch_size*patch_size)
    def forward(self,x):
        batch=x.shape[0]
        h=self.img_size//self.patch_size
        w=self.img_size//self.patch_size
        x=self.proj(self.norm(x))
        x=x.view(batch,h,w,self.channel,self.patch_size,self.patch_size)
        x=x.permute(0,3,1,4,2,5).contiguous()
        x=x.view(batch,self.channel,self.img_size,self.img_size)
        return x

class LDT(nn.Module):
    def __init__(self, emb_dim,con_dim,channel=4,img_size=32,patch_size=2,hidden_cha=768,device='cpu'):
        super().__init__()
        self.patchfy=Patchfyemb(channel,img_size,patch_size,hidden_cha)
        self.time_emb=TimeEmbeddings(emb_dim,device)
        self.con_dim=nn.Embedding(con_dim+1,emb_dim) #类别索引转化为embed，不能用linear 保留额外索引作空索引
        self.dit=DiT(emb_dim,emb_dim)
        self.norm=nn.LayerNorm(hidden_cha)
        self.unpatchfy=Unpatchfy(channel,img_size,patch_size,hidden_cha)
        self.NULL_IDX=con_dim
        self.device=device
    def forward(self,x,t,cond=None):
        B=x.shape[0]
        x=self.patchfy(x)
        time_emb=self.time_emb(t)
        cond=torch.full((B,),self.NULL_IDX,dtype=torch.long,device=self.device) if torch.rand(1).item() < 0.2 else cond
        con_emb=self.con_dim(cond)
        con=con_emb+time_emb
        x=self.dit(x,con)
        x=self.norm(x)
        x=self.unpatchfy(x)
        return x
