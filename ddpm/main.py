from math import inf
from pdb import run
import torch
import torch.optim as optim
from dataset import get_loaders
from ddpmnet import UNet,DDPMNet
from trainer import train_epoch
from visualize import visualize_samples,visualize_loss_curve,visualize_denoising_process
from evaluate import evaluate
import time
import os

def main():
    #超参数
    device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'device is {device}')

    num_epoch=10
    time_step=1000
    lr=1e-4
    model_channel=64

    os.chdir(os.path.dirname(__file__)) #工作路径切换到当前文件所在的目录（ddpm文件夹）
    save_path='results'
    run_dir=os.path.join(save_path,f'run_{int(time.time())}')
    os.makedirs(run_dir,exist_ok=True)
    run_dir_modals=os.path.join(run_dir,'modals')
    os.makedirs(run_dir_modals,exist_ok=True)
    run_dir_visualize=os.path.join(run_dir,'visualize')
    os.makedirs(run_dir_visualize,exist_ok=True)
    run_dir_evaluate=os.path.join(run_dir,'evaluate')
    os.makedirs(run_dir_evaluate,exist_ok=True) 

    #准备model
    train_loader,val_loader,test_loader=get_loaders()
    print("loaders get ready")
    unet=UNet(in_channel=1,model_channel=model_channel,num_res_block=2).to(device)
    ddpmnet=DDPMNet(unet,timesteps=time_step,device=device)
    print("model get ready")
    optimizier=optim.AdamW(ddpmnet.parameters(),lr=lr)  #在此之前要把模型移动到device上
    scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizier,T_max=time_step)
    print("optimizier and scheduler get ready")

    train_loss=[]
    val_loss=[]
    best_loss=float(inf)

    print("Start training...")
    for epoch in range(num_epoch):
        train_epoch_loss=train_epoch(ddpmnet,train_loader,optimizier,device,epoch)
        print(f'train_loss in EPOCH{epoch} is {train_epoch_loss:.6f}')

        val_epoch_loss=evaluate(ddpmnet,val_loader,device,epoch,True,save_path=run_dir_evaluate)
        print(f'val_loss in EOOCH{epoch} is {val_epoch_loss:.6f}')

        scheduler.step()  #更新学习率

        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        if val_epoch_loss<best_loss:
            best_loss=val_epoch_loss
            torch.save(unet.state_dict(),os.path.join(run_dir_modals,'best_model.pth'))

    print(f'Finish training! The best loss is{best_loss:.6f}')
    visualize_loss_curve(train_loss,val_loss,os.path.join(run_dir_visualize,'loss_curve.png'))
    #加载最优模型评估
    unet.load_state_dict(torch.load(os.path.join(run_dir_modals,'best_model.pth')))
    DDPM=DDPMNet(unet,timesteps=time_step,device=device)
    
    visualize_denoising_process(DDPM,device,os.path.join(run_dir_visualize,'denoising_process.png'))
    visualize_samples(DDPM,os.path.join(run_dir_visualize,'samples.png'))
    
    print("visualization in results/run_xxx/visualize")

if __name__=='__main__':
    main()
