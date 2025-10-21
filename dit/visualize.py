from networkx import descendants
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import make_grid
import tqdm

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

def visualize_vae(vae,dataloader,ncol=8,save_path=None,title='comparison x,x_encoder,x_decoder',device='cpu'):
    '''比较vae的x x_encoder x_decoder'''
    vae.eval()
    x,_=next(iter(dataloader))
    x=x[:ncol].to(device)
    x_decoder,x_encoder=vae(x)
    grid=torch.cat(x,x_encoder,x_decoder,dim=0)
    grid=make_grid(grid,nrow=ncol).detach().cpu().numpy()
    plt.title(title)
    plt.imshow(grid)
    plt.axis(emit=False)

    if save_path:
        _safe_savefig(save_path, default_name='denoising.png',dpi=150, bbox_inches='tight')
    
    plt.close()

def visualize_ldt(vae,ldt,ddpm,channels=4,num_classes=10,samples_perClass=2,img_size=32,
                  device='cpu',save_path=None,title='visualize sample'):
    '''图像的生成过程（从潜空间开始去噪 后decoder)'''
    vae.eval()
    ldt.eval()
    with torch.no_grad():
        imgs=[]
        steps_to_show=reversed(list[np.linspace(0,ddpm.timesteps,dtype=int)])
        batch=num_classes*samples_perClass
        labels=torch.arange(num_classes,device=device).repeat_interleave(samples_perClass)
        z=torch.randn(batch,channels,img_size,img_size,device=device)
        loop=tqdm(reversed(range(ddpm.timesteps)),desc='denoising')
        for i in loop:
            t_tensor=torch.full((batch,),i,dtype=torch.long,device=device)
            z=ddpm.p_sample(z,t_tensor,labels)
            if i in steps_to_show:
                z_decoder,_=vae(z[0])
                imgs.append(z_decoder)
    
    fig,axe=plt.subplots(1,steps_to_show,figsize=(steps_to_show*3,3))
    for i,img in enumerate(imgs):
        axe[i]=img.detach().cpu().numpy()
        if img[0]==1:
            axe[i].imshow(img.sequeeze())
        else:
            axe[i].imshow(np.transpose(img,(1,2,0)))
        axe[i].axis('off')

    _safe_savefig(save_path,fig,title)
    plt.title(title)
    plt.close()

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