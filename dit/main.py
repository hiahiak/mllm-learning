from math import fabs, inf
from pdb import run
import torch
import torch.optim as optim
from dataset import get_loaders
from ddpm import DDPMNet
from ldt import LDT
from vae import VAE
from trainer import train_epoch,vae_train_epoch
from evaluate import evaluate
from visualize import visualize_ldt,visualize_vae,visualize_loss_curve
import time
import os
'''ldt里面没有将数据统一传到device 需修改 并在调用时传入device;  条件label处理'''
def main():
    #超参数
    device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f'device is {device}')

    num_epoch=10
    time_step=1000
    lr=1e-4
    hidden_channel=768
    emb_dim=768
    con_dim=10

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
    vae=VAE().to(device)
    ldt=LDT(emb_dim,con_dim,device=device).to(device)
    ddpm=DDPMNet(ldt,device=device).to(device)
    
    print("model get ready")
    optimizier_ddpm=optim.AdamW(vae.parameters(),lr=lr)  #在此之前要把模型移动到device上
    scheduler_ddpm=optim.lr_scheduler.CosineAnnealingLR(optimizier_ddpm,T_max=time_step)
    optimizier_vae=optim.AdamW(vae.parameters(),lr=lr)  #在此之前要把模型移动到device上
    scheduler_vae=optim.lr_scheduler.CosineAnnealingLR(optimizier_vae,T_max=time_step)

    print("optimizier and scheduler get ready")

    train_loss_ddpm=[]
    val_loss_ddpm=[]
    best_loss_ddpm=float(inf)
    train_loss_vae=[]
    val_loss_vae=[]
    best_loss_vae=float(inf)

    print("Start training vae...")
    for epoch in range(num_epoch):
        train_epoch_loss_vae=vae_train_epoch(vae,train_loader,optimizier_vae,device,epoch)
        print(f'train_loss in EPOCH{epoch} is {train_epoch_loss_vae:.6f}')

        val_epoch_loss_vae=evaluate(vae,val_loader,device,epoch,True,save_path=run_dir_evaluate)
        print(f'val_loss in EOOCH{epoch} is {val_epoch_loss_vae:.6f}')

        scheduler_vae.step()  #更新学习率

        train_loss_vae.append(train_epoch_loss_vae)
        val_loss_vae.append(val_epoch_loss_vae)
        if val_epoch_loss_vae<best_loss_vae:
            best_loss_vae=val_epoch_loss_vae
            torch.save(ldt.state_dict(),os.path.join(run_dir_modals,'best_vae_model.pth'))

    print(f'Finish training vae! The best loss is{best_loss_vae:.6f}')
    visualize_loss_curve(train_loss_vae,val_loss_vae,os.path.join(run_dir_visualize,'vae_loss_curve.png'))

    print("Start training ddpm...")
    for epoch in range(num_epoch):
        train_epoch_loss_ddpm=train_epoch(ddpm,train_loader,optimizier_ddpm,device,epoch,vae)
        print(f'train_loss in EPOCH{epoch} is {train_epoch_loss_ddpm:.6f}')

        val_epoch_loss_ddpm=evaluate(ddpm,val_loader,device,epoch,False,save_path=run_dir_evaluate,ddpm=True)
        print(f'val_loss in EOOCH{epoch} is {val_epoch_loss_ddpm:.6f}')

        scheduler_ddpm.step()  #更新学习率

        train_loss_ddpm.append(train_epoch_loss_ddpm)
        val_loss_ddpm.append(val_epoch_loss_ddpm)
        if val_epoch_loss_ddpm<best_loss_ddpm:
            best_loss_ddpm=val_epoch_loss_ddpm
            torch.save(ldt.state_dict(),os.path.join(run_dir_modals,'best_ddpm_model.pth'))

    print(f'Finish training ddpm! The best loss is{best_loss_ddpm:.6f}')
    visualize_loss_curve(train_loss_ddpm,val_loss_ddpm,os.path.join(run_dir_visualize,'ddpm_loss_curve.png'))
    #加载最优模型评估
    ldt.load_state_dict(torch.load(os.path.join(run_dir_modals,'best_ddpm_model.pth')))
    vae.load_state_dict(torch.load(os.path.join(run_dir_modals,'best_vae_model.pth')))
    DDPM=DDPMNet(ldt,timesteps=time_step,device=device)
    visualize_vae(vae,test_loader,device=device)
    visualize_ldt(vae,ldt,DDPM,device=device)
    print("visualization in results/run_xxx/visualize")

if __name__=='__main__':
    main()
