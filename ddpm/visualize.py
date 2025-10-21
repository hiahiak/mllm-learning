import torch
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']  #使用黑体或默认字体
plt.rcParams['axes.unicode_minus']=False  #显示负号

def _safe_savefig(save_path, fig=None, default_name='figure.png', **save_kwargs):
    """
    如果 save_path 是目录，则在目录下创建 default_name；
    如果是文件路径，则按该路径保存；
    忽略 None 或 空字符串。
    """
    if not save_path:
        return
    fig = fig or plt.gcf()
    # 若用户传入的是已存在目录或以路径分隔符结尾，视为目录
    if os.path.isdir(save_path) or save_path.endswith(os.sep):
        save_dir = save_path
        save_fp = os.path.join(save_dir, default_name)
    else:
        save_dir = os.path.dirname(save_path) or '.'
        save_fp = save_path
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(save_fp, **save_kwargs)
    print(f"Saved to: {save_fp}")
    
def visualize_samples(ddpm,save_path=None,num_samples=16,title='ddpm generated sample',epoch=None):
    '''可视化ddpm的生成'''
    ddpm.model.eval()  #关闭训练行为：dropout、norm等
    with torch.no_grad():
        samples=ddpm.sample(num_samples,img_size=28)
        samples=((samples+1.0)/2.0).cpu()
    #创建图像网格，自动调整大小
    grid_cols=int(np.ceil(np.sqrt(num_samples)))
    grid_rows=int(np.ceil(num_samples/grid_cols))
    fig,axes=plt.subplots(grid_rows,grid_cols,figsize=(grid_rows*2,grid_cols*2))
    fig.suptitle(f'{title}'+(f'-EPOCH{epoch}' if epoch is not None else ' '),fontsize=16)

    for i in range(grid_cols*grid_rows):
        row,col=divmod(i,grid_cols)
        ax=axes[row,col] if grid_rows>1 else axes[col] if grid_cols>1 else axes[row]  #根据网格维度自动适配

        if i<num_samples:
            img=samples[i].detach().cpu().numpy()
            if img.shape[0]==1:  #灰度图 
                ax.imshow(img.squeeze(),cmap='gray')
            else: #RGB
                ax.imshow(np.transpose(img,(1,2,0))) #imshow支持(H,W,C)
    plt.tight_layout()  #自动调整子图间距
    plt.subplots_adjust(top=0.92)  #预留位置给title

    if save_path:
        _safe_savefig(save_path, fig, default_name='samples.png',dpi=150, bbox_inches='tight')
    
    plt.close()
    ddpm.model.train()

def visualize_denoising_process(ddpm,device,save_path=None,epoch=None,
                                title='ddpm denoising process',stepToShow=8):
    '''可视化ddpm的去噪过程'''
    ddpm.model.eval()
    with torch.no_grad():
        img=torch.randn(1,1,28,28,device=device)  #从纯噪声开始
        imgs=[]
        timestepsToShow=np.linspace(0,ddpm.timesteps-1,stepToShow,dtype=int)
        #逐步去噪并保存中间结果
        for i in reversed(range(ddpm.timesteps)):
            t=torch.full((1,),i,dtype=torch.long).to(device)
            img=ddpm.p_sample(img,t)
            if i in timestepsToShow:
                img=(img+1.0)/2.0  #反归一化到(0,1)
                imgs.append(img.detach().cpu().numpy())
        #创建可视化
    fig,axes=plt.subplots(1,stepToShow,figsize=(stepToShow*2,2))
    fig.suptitle(f'{title}'+(f'-EPOCH{epoch}' if epoch is not None else ' '),fontsize=16)
    for i,img in enumerate(imgs):
        ax=axes[i] if stepToShow>1 else axes
        if img.shape[0]==1:  #灰度图
            ax.imshow(img.squeeze(),cmap='gray')
        else: #RGB
            ax.imshow(np.transpose(img,(1,2,0))) #imshow支持(H,W,C)
        ax.set_title(f'T={timestepsToShow[i]}')
        ax.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        _safe_savefig(save_path, fig, default_name='denoising.png',dpi=150, bbox_inches='tight')
    
    plt.close()
    ddpm.model.train()

def compare_original_noise_denoised(ddpm,dataloader,device,save_path=None,epoch=None,
                                    title='original vs denoised'):
    '''对比原始图像、加噪图像与去噪结果   '''
    ddpm.model.eval()
    #获取原始图像
    data_iter=iter(dataloader)
    real_data,_=next(data_iter)
    real_data=real_data[:6].to(device)  #取前6张
    real_data=(real_data*2.0-1.0)  #归一化到(-1,1)

    timesteps=[200,500,800]  #选择不同的加噪时间步
    fig,axes=plt.subplots(len(timesteps),6*3,figsize=(18,3*len(timesteps)))
    fig.suptitle(f'{title}'+(f'-EPOCH{epoch}' if epoch is not None else ' '),fontsize=16)

    for row,t in enumerate(timesteps):
        t_batch=torch.full((real_data.size(0),),t,dtype=torch.long).to(device)
        noise=torch.randn_like(real_data).to(device)
        noisy_data=ddpm.q_sample(real_data,t_batch,noise)  #加噪
        predicted_noise=ddpm.model(noisy_data,t_batch)  #预测噪声for后续简化运算
        #简化 一步预测
        alpha_t = ddpm.alpha_cumprod[t_batch].reshape(-1, 1, 1, 1)
        beta_t = (1 - alpha_t).reshape(-1, 1, 1, 1)
        predicted_clean = (noisy_data - torch.sqrt(beta_t) * predicted_noise) / torch.sqrt(alpha_t)

        for i in range(6):
            col_base=i*3
            #原始图像
            ax=axes[row,col_base] if len(timesteps)>1 else axes[col_base]
            img=real_data[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title('Original' if row==0 else '')
            ax.axis('off')

            #加噪图像
            ax=axes[row,col_base+1] if len(timesteps)>1 else axes[col_base+1]
            img=noisy_data[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title(f'Noisy T={t}' if row==0 else '')
            ax.axis('off')

            #去噪图像
            ax=axes[row,col_base+2] if len(timesteps)>1 else axes[col_base+2]
            img=predicted_clean[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title('Denoised' if row==0 else '')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  #预留位置给title
    if save_path:
        _safe_savefig(save_path, fig, default_name='orig_noisy_denoised.png',dpi=150, bbox_inches='tight')
    plt.close()
    ddpm.model.train()

def visualize_loss_curve(train_loss,val_loss,save_path=None,title='Training Loss Curve'):
    '''可视化训练损失曲线'''
    epoch=range(1,len(train_loss)+1)
    _,ax=plt.subplots(1,1,figsize=(6,4))
    ax.plot(epoch,train_loss,'r-',label='Train Loss')
    if val_loss:
        ax.plot(epoch,val_loss,'b-',label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()  #显示图例
    ax.grid(True, linestyle='--', alpha=0.5)  #添加网格

    #添加min loss点
    min_train_idx=np.argmin(train_loss)
    ax.plot(min_train_idx+1,train_loss[min_train_idx],'ro')
    min_val_idx=np.argmin(val_loss) if val_loss else None
    if min_val_idx is not None:
        ax.plot(min_val_idx+1,val_loss[min_val_idx],'bo')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  #预留位置给title

    if save_path:
        _safe_savefig(save_path, plt.gcf(), default_name='loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

def visualize_interpolation(ddpm,device,save_path=None,time_step=10):
    '''可视化潜在空间插值'''
    ddpm.model.eval()
    with torch.no_grad():
        #随机生成两个潜在向量,一起点一终点
        z1=torch.randn(1,1,28,28,device=device)
        z2=torch.randn(1,1,28,28,device=device)
        alphas=np.linspace(0,1,10)  #插值系数
        interpolated_imgs=[]
        for alpha in alphas:
            z=alpha*z1+(1-alpha)*z2
            #从时间步time_step开始去噪
            img=z
            for i in reversed(range(time_step)):
                t=torch.full((1,),i,dtype=torch.long).to(device)
                img=ddpm.p_sample(img,t)
            img=(img+1.0)/2.0  #反归一化到(0,1)
            interpolated_imgs.append(img.detach().cpu().numpy())
        
    fig,axes=plt.subplots(1,len(alphas),figsize=(len(alphas)*2,2))
    fig.suptitle(f'Latent Space Interpolation from T={time_step}',fontsize=16)
    for i,img in enumerate(interpolated_imgs):
        ax=axes[i] if len(alphas)>1 else axes
        if img.shape[0]==1:  #灰度图
            ax.imshow(img.squeeze(),cmap='gray')
        else: #RGB
            ax.imshow(np.transpose(img,(1,2,0))) 
        ax.set_title(f'α={alphas[i]:.2f}')
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  #预留位置给title

    if save_path:
        _safe_savefig(save_path, fig, default_name='interpolation.png',dpi=150, bbox_inches='tight')
    plt.close()
    ddpm.model.train()

def visualize_completely_denoising(ddpm,dataloader,device,time_max=500,save_path=None,epoch=None,
                                    title='visualize completely denoising process'):
    '''可视化完整去噪过程的对比'''
    print(f'Visualizing completely denoising process at T={time_max}...')
    ddpm.model.eval()
    #获取原始图像
    data_iter=iter(dataloader)
    real_data,_=next(data_iter)
    real_data=real_data[:3].to(device)  #取前3张
    real_data=(real_data*2.0-1.0)  #归一化到(-1,1)

    timesteps=[200,time_max]  #选择不同的加噪时间步
    fig,axes=plt.subplots(len(timesteps),3*3,figsize=(9,3*len(timesteps)))
    fig.suptitle(f'{title}'+(f'-EPOCH{epoch}' if epoch is not None else ' '),fontsize=16)

    for row,t in enumerate(timesteps):
        t_batch=torch.full((real_data.size(0),),t,dtype=torch.long).to(device)
        noise=torch.randn_like(real_data).to(device)
        noisy_data=ddpm.q_sample(real_data,t_batch,noise)  #加噪
        
        #完整denoising
        denoised_data=noisy_data.clone()
        for i in reversed(range(t+1)):
            t_i=torch.full((real_data.size(0),),i,dtype=torch.long).to(device)
            denoised_data=ddpm.p_sample(denoised_data,t_i)

        for i in range(real_data.size(0)):
            col_base=i*3
            #原始图像
            ax=axes[row,col_base] if len(timesteps)>1 else axes[col_base]
            img=real_data[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title('Original' if row==0 else '')
            ax.axis('off')

            #加噪图像
            ax=axes[row,col_base+1] if len(timesteps)>1 else axes[col_base+1]
            img=noisy_data[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title(f'Noisy T={t}' if row==0 else '')
            ax.axis('off')

            #去噪图像
            ax=axes[row,col_base+2] if len(timesteps)>1 else axes[col_base+2]
            img=denoised_data[i].detach().cpu().numpy()
            if img.shape[0]==1:
                ax.imshow(img.squeeze(),cmap='gray')
            else:
                ax.imshow(np.transpose(img,(1,2,0)))
            ax.set_title('Denoised' if row==0 else '')
            ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  #预留位置给title
    if save_path:
        _safe_savefig(save_path, fig, default_name='complete_denoising.png',dpi=150, bbox_inches='tight')
    plt.close()
    ddpm.model.train()
