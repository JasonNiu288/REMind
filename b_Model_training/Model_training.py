import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange 
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from timm.utils import ModelEmaV3 
from tqdm import tqdm 
import matplotlib.pyplot as plt 
import torch.optim as optim
import numpy as np
import os
import tifffile as tiff

from a_Networks.Networks import UNET
from c_utilities.utilities import Stage1_Dataset, SG2_3D_StackDataset, reverse_sampling,\
                                  monitoring_SG1_training, DDPM_Scheduler,monitoring_SG2_training



def train_model(Channels: list = None,
          Attentions = None,
          Upscales = None,
          num_groups = 32,
          batch_size: int=64,
          img_size: int=64,
          input_channels: int=3,
          sf: int = 4,
          num_time_steps: int=1000,
          num_epochs: int=15,
          ema_decay: float=0.9999,  
          lr=2e-5,
          checkpoint_path_SG1: str=None,
          checkpoint_path_SG2: str=None,
          load_pretrained: str=False,
          load_pretrained_epoch: int=192,
          train_dataset_path_SG1: str='./',
          test_dataset_path_SG1: str='./',
          train_dataset_path_SG2: str='./',
          test_dataset_path_SG2: str='./',
          monitor_process_path: str='./',
          N_mean: int=1,
          L: int=6,
          REMind_mode: str='SG1'):

    if REMind_mode == 'SG1':
        train_dataset = Stage1_Dataset(train_dataset_path_SG1,sf=sf, img_size=img_size)
        input_channels = 3  #time t, input data, low-res condition
    elif REMind_mode == 'SG2':    
        train_dataset = SG2_3D_StackDataset(train_dataset_path_SG2)
        input_channels = 5 #4 from embedding_total and 1 from input
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    if REMind_mode== 'SG1':
        model = UNET(
            Channels=Channels,
            Attentions=Attentions,
            Upscales=Upscales,
            num_groups=num_groups,
            dropout_prob=0.1,
            num_heads=8,
            input_channels=input_channels,
            output_channels=1,
            time_steps=num_time_steps,
            img_size=img_size,
            REMind_mode=REMind_mode
        ).cuda()
    elif REMind_mode== 'SG2':
            model = UNET(
            Channels=Channels,
            Attentions=Attentions,
            Upscales=Upscales,
            num_groups=num_groups,
            dropout_prob=0.1,
            num_heads=8,
            input_channels=input_channels,
            output_channels=1,
            time_steps=num_time_steps,
            img_size=img_size,
            L=L,  
            REMind_mode=REMind_mode
        ).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    
    if load_pretrained:
        checkpoint = torch.load(checkpoint_path_SG1 + f'/model_{load_pretrained_epoch}.pth')
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    L1 = nn.L1Loss(reduction='mean')  

    train_loader_list = list(train_loader)
    for i in range(num_epochs):
        total_loss = 0
        total_loss_ref = 0

        for bidx, batch in enumerate(tqdm(train_loader_list, desc=f"Epoch {i+1}/{num_epochs}")):
            if REMind_mode == 'SG1':
                x_lr, x_hr = batch  # Unpack both low-res and high-res images
                x_lr = x_lr.cuda()
                x_hr = x_hr.cuda()
                #Upsampling x_lr to concatenate with hr as the condition (batch_size, 1, H, W)
                x_lr_up = F.interpolate(x_lr, size=(img_size, img_size), mode='bicubic', align_corners=False)
                t = torch.randint(0,num_time_steps,(batch_size,))
                e = torch.randn_like(x_hr, requires_grad=False)
                a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
                x_hr = (torch.sqrt(a)*x_hr) + (torch.sqrt(1-a)*e)
            
            elif REMind_mode == 'SG2':
                x, _ = batch  # Unpack only the first element, ignoring the second
                x = x.cuda()
                top = x[:, 0, :, :].unsqueeze(dim=1) #0 channel as condition 1
                bottom = x[:, L-1, :, :].unsqueeze(dim=1) #1 channel as condition 2

                t = torch.randint(0,num_time_steps,(batch_size,))
                valid_channels = torch.tensor([i for i in range(1, L) if i not in (0, L-1)])  #0 and 3 refer the conditional channel idx
                c_idx = valid_channels[torch.randint(0, len(valid_channels), (batch_size,))]
                x = x[torch.arange(x.size(0)), c_idx, :, :].unsqueeze(dim=1)
                
                e = torch.randn_like(x, requires_grad=False)
                a = scheduler.alpha[t].view(batch_size,1,1,1).cuda()
                x = (torch.sqrt(a)*x) + (torch.sqrt(1-a)*e)

            if REMind_mode == 'SG1':
                output = model(x_hr, t, lr_up=x_lr_up)
            if REMind_mode == 'SG2':  
                output = model(x, t, c_idx=c_idx, top=top, bottom=bottom)
  
            optimizer.zero_grad()
            loss = L1(output, e)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)  
    
        print(f'Epoch {i+1} | Loss {(total_loss / len(train_loader)):.6f}')

        checkpoint = {
            'weights': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'ema': ema.state_dict()
        }

        if i%200 ==0:
            if REMind_mode == 'SG1':
                checkpoint_path_epoch = os.path.join(checkpoint_path_SG1, f'model_{i}.pth')
                monitoring_SG1_training(test_dataset_path_SG1, monitor_process_path, img_size, sf, model, scheduler, num_time_steps, i, N_mean, REMind_mode)
            if REMind_mode == 'SG2':
                checkpoint_path_epoch = os.path.join(checkpoint_path_SG2, f'model_{i}.pth')
                monitoring_SG2_training(test_dataset_path_SG2, monitor_process_path, img_size, L, model, scheduler, num_time_steps, i, N_mean, REMind_mode)
            torch.save(checkpoint, checkpoint_path_epoch)
            