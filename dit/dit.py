import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaLNZero(nn.Module):
    '''输入con_dim,输出作用于主干线transformer上(emb_dim) 所以设计维度转换'''
    def __init__(self, emb_dim,con_dim):
        super().__init__()
        self.norm=nn.LayerNorm(emb_dim)
        self.mlp=nn.Sequential(
            nn.Linear(con_dim,4*con_dim),
            nn.SiLU(),
            nn.Linear(4*con_dim,4*emb_dim),
            nn.SiLU(),
            nn.Linear(4*emb_dim,3*emb_dim)
        )
    def forward(self,x,cond):
        x=self.norm(x)
        cond=self.mlp(cond)
        scale,shift,gate=cond.chunk(3,dim=-1)
        scale=scale.unsqueeze(1)
        shift=shift.unsqueeze(1)
        gate=gate.unsqueeze(1)
        return x* (1+scale) + shift,gate

class FFN(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.mlp=nn.Sequential(
            nn.Linear(emb_dim,4*emb_dim),
            nn.SiLU(),
            nn.Linear(4*emb_dim,emb_dim)
        )
        self.dropout=nn.Dropout(0.1)
    def forward(self,x):
        return self.dropout(self.mlp(x))

class DiTBlock(nn.Module):
    def __init__(self,emb_dim,con_dim):
        super().__init__()
        self.adaln1=AdaLNZero(emb_dim,con_dim)
        self.adaln2=AdaLNZero(emb_dim,con_dim)
        self.ffn=FFN(emb_dim)
        self.mha=nn.MultiheadAttention(emb_dim,num_heads=8,batch_first=True)
    def forward(self,x,cond):
        h,gate1=self.adaln1(x,cond)
        h, _ = self.mha(h, h, h)
        x=x+gate1*h
        h,gate2=self.adaln2(x,cond)
        h=self.ffn(h)
        x=x+gate2*h
        return x
    
class DiT(nn.Module):
    def __init__(self, emb_dim,con_dim,depth=8):
        super().__init__()
        self.layers=nn.ModuleList([
            DiTBlock(emb_dim,con_dim) for _ in range(depth)
        ])
    def forward(self,x,cond):
        for layer in self.layers:
            x=layer(x,cond)
        return x
    