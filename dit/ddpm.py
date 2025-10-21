import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPMNet(nn.Module):
    def __init__(self,model, beta_start=1e-4, beta_end=2e-2, timesteps=1000, device='cpu' ):
        '''没有可训练的参数  参数在model中  创建的tensor都要to(device)'''
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

    def p_sample(self, x_t, t,cond=None):
        '''反向去噪过程'''
        with torch.no_grad():

            predicted_noise = self.model(x_t, t,cond)
            coef1 = 1 / torch.sqrt(self.alpha[t]).view(-1, 1, 1, 1)
            coef2 = (self.beta[t] / torch.sqrt(1 - self.alpha_cumprod[t])).view(-1, 1, 1, 1)
            mean = coef1 * (x_t - coef2 * predicted_noise)
            if t[0] == 0:
                return mean
            else:
                noise = torch.randn_like(x_t).to(self.device)
                variance = self.posterior_variance[t].view(-1, 1, 1, 1)
                return mean + torch.sqrt(variance) * noise
    
    def sample(self,batch_size,img_size=28,cond=None):
        '''从纯噪声生成图像'''
        self.model.eval()
        img = torch.randn(batch_size, 1, img_size, img_size).to(self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long).to(self.device)  #每一个时间步都对一个batch采样
            img = self.p_sample(img, t)
        return img
    
    def compute_loss(self, x_start,cond=None):
        '''计算损失函数'''
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()  #对batch里每个样本都抽取一个t
        noise = torch.randn_like(x_start).to(self.device)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.model(x_noisy,t,cond)
        return F.mse_loss(noise, predicted_noise)