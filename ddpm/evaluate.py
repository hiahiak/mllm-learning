import torch
from tqdm import tqdm  
from visualize import visualize_denoising_process,visualize_loss_curve,visualize_samples,compare_original_noise_denoised
import os

def evaluate(ddpm,dataloader,device,epoch=None,visualize=False,save_path=None):
    ddpm.model.eval()
    eval_loss=[]
    running_loss=0.0
    num_batches=len(dataloader)
    loop=tqdm(dataloader,desc=f'Evaluate Epoch:{epoch+1}',leave=False)  

    with torch.no_grad():
        for batch_idx,(data,_) in enumerate(loop):
            data=data.to(device)
            data=data*2.0-1.0  
            
            loss=ddpm.compute_loss(data)    
            eval_loss.append(loss.item())
            running_loss+=loss.item()

            loop.set_postfix({
                'loss':f'{loss.item():.6f}',
                'avg_loss':f'{running_loss/(batch_idx+1):.6f}'
            })
    avg_loss=running_loss/num_batches

    #可视化 
    if visualize and epoch is not None:
        if save_path is None:
            save_path = 'results'
        os.makedirs(save_path, exist_ok=True)

        img_size = data.size(2)
        # 1. 可视化去噪过程 -> 保存为文件
        denoise_fp = os.path.join(save_path, f'denoising_epoch{epoch}_size{img_size}.png')
        visualize_denoising_process(ddpm, device, denoise_fp, epoch, title=f'denoising_epoch{epoch}', stepToShow=8)

        # 2. 可视化损失曲线
        loss_fp = os.path.join(save_path, f'loss_epoch{epoch}.png')
        visualize_loss_curve(eval_loss, None, loss_fp, title=f'Loss_epoch{epoch}')

        # 3. 可视化生成样本
        samples_fp = os.path.join(save_path, f'samples_epoch{epoch}_size{img_size}.png')
        visualize_samples(ddpm, samples_fp, num_samples=16, title=f'samples_epoch{epoch}')

        # 4. 对比原始/加噪/去噪图像
        compare_fp = os.path.join(save_path, f'orig_noisy_denoised_epoch{epoch}_size{img_size}.png')
        compare_original_noise_denoised(ddpm, dataloader, device, compare_fp, epoch, title=f'orig_noisy_denoised_epoch{epoch}')
    
    return avg_loss