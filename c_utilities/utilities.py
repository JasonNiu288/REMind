import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import os
import tifffile as tiff
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from timm.utils import ModelEmaV3 
from torch.utils.data import Dataset, DataLoader



class Stage1_Dataset(Dataset):
    def __init__(self, stacks_path, sf, img_size):
        self.slices = [os.path.join(stacks_path, file) 
               for file in sorted(os.listdir(stacks_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        self.sf = sf
        self.img_size = img_size
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slices = tiff.imread(self.slices[idx])
        slice_tensor = torch.tensor(slices, dtype=torch.float32) / 255.0
        HR_img = slice_tensor[0:self.img_size, 0:self.img_size].unsqueeze(dim=0)
        LR_img = slice_tensor[0:int(self.img_size/self.sf), self.img_size:(self.img_size + int(self.img_size/self.sf))].unsqueeze(dim=0)


        return LR_img, HR_img

# Dataset for 3D Image stack at SG1 (only for sparse FIB-SEM to sparse super-resolution)
class LHR_3D_Dataset(Dataset):
    def __init__(self, stacks_path, sf, img_size):
        self.slices = [os.path.join(stacks_path, file) 
               for file in sorted(os.listdir(stacks_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))]
        self.sf = sf
        self.img_size = img_size
    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slices = tiff.imread(self.slices[idx])
        slice_tensor = torch.tensor(slices, dtype=torch.float32) / 255.0
        HR_img = slice_tensor[:, 0:self.img_size, 0:self.img_size].unsqueeze(dim=0)
        LR_img = slice_tensor[:, 0:int(self.img_size/self.sf), self.img_size:(self.img_size + int(self.img_size/self.sf))].unsqueeze(dim=0)


        return LR_img, HR_img


class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps: int=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        return self.beta[t], self.alpha[t]



class Low_res_img_and_Time_Step_Embeddings(nn.Module):
    def __init__(self, time_steps:int, img_size:int):
        super().__init__()
        self.embed_dim = int(img_size * img_size)
        position_t = torch.arange(time_steps).unsqueeze(1).float()
        div_t = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
        embeddings_t = torch.zeros(time_steps, self.embed_dim, requires_grad=False) # for time step t
        embeddings_t[:, 0::2] = torch.cos(position_t * div_t)
        embeddings_t[:, 1::2] = torch.sin(position_t * div_t) 
        
        self.t_embeddings = embeddings_t

    def forward(self, x, t, lr_up):
         
        embeds_t = self.t_embeddings[t].reshape(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        embeds_total = torch.cat((embeds_t, lr_up), dim =1).to(x.device)
        
        return embeds_total

class slicesEmbeddings(nn.Module):
    def __init__(self, time_steps:int, L:int, img_size:int):
        super().__init__()
        self.L = L
        self.embed_dim = int(img_size * img_size)
        position_c= torch.arange(L).unsqueeze(1).float()
        position_t = torch.arange(time_steps).unsqueeze(1).float()
        div_t = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
        div_c = torch.exp(torch.arange(0, self.embed_dim, 2).float() * -(math.log(10000.0) / self.embed_dim))
        embeddings_c = torch.zeros(L, self.embed_dim, requires_grad=False) #for channel position
        embeddings_t = torch.zeros(time_steps, self.embed_dim, requires_grad=False) # for time step t
        embeddings_c[:, 0::2] = torch.cos(position_c * div_c)  #position*div: (1000,1) x (1, 512) = (1000,512)
                                                         # starts from index 0 and uses a step of 2, effectively selecting every second column.
        embeddings_c[:, 1::2] = torch.sin(position_c * div_c)  # likewise, start from index 0, uses a step of 2, effectively selecting every second column.
        
        embeddings_t[:, 0::2] = torch.cos(position_t * div_t)
        embeddings_t[:, 1::2] = torch.sin(position_t * div_t) 
        
        self.channel_position_embeddings = embeddings_c
        self.t_embeddings = embeddings_t

    def forward(self, x, t, c_idx, top, bottom):
         
        embeds_t = self.t_embeddings[t].reshape(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        embeds_c = self.channel_position_embeddings[c_idx].reshape(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
        embeds_total = torch.cat((embeds_t, embeds_c, top, bottom), dim =1).to(x.device)
        
        return embeds_total


# Read 3D stack, e.g., 6x128x128, for SG2 training and test
class SG2_3D_StackDataset(Dataset):
    def __init__(self, stacks_path):
        self.slices = [os.path.join(stacks_path, file) 
               for file in sorted(os.listdir(stacks_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        slices = tiff.imread(self.slices[idx])
        slice_3D = torch.tensor(slices, dtype=torch.float32) / 255

        return[slice_3D, 0]



def reverse_sampling(model, scheduler, num_time_steps, img_size, batch_size, REMind_mode, **kwargs):
    if REMind_mode == 'SG1':
        device = kwargs['lr_up'].device
    elif REMind_mode == 'SG2':
        device = kwargs['top'].device
    else:
        raise ValueError("Invalid mode selected. Choose 'SG1' or 'SG2'.")

    z = torch.randn(batch_size, 1, img_size, img_size).to(device)

    for t in reversed(range(1, num_time_steps)):
        t = torch.full((batch_size,), t)
        temp = (scheduler.beta[t].view(batch_size,1,1,1) /
               ((torch.sqrt(1 - scheduler.alpha[t].view(batch_size,1,1,1))) *
                (torch.sqrt(1 - scheduler.beta[t].view(batch_size,1,1,1))))).to(device)
        
        if REMind_mode == 'SG1':
            z = (1 / (torch.sqrt(1 - scheduler.beta[t].view(batch_size,1,1,1)).to(device))) * z - \
                (temp * model(z, t, lr_up=kwargs['lr_up']))
        elif REMind_mode == 'SG2':
            z = (1 / (torch.sqrt(1 - scheduler.beta[t].view(batch_size,1,1,1)).to(device))) * z - \
                (temp * model(z, t, c_idx=kwargs['c_idx'], top=kwargs['top'], bottom=kwargs['bottom']))

        e = torch.randn(1, 1, img_size, img_size).to(device)
        z = z + (e * torch.sqrt(scheduler.beta[t].view(batch_size,1,1,1).to(device)))

    t_0 = torch.full((batch_size,), 0)
    temp = scheduler.beta[t_0].view(batch_size,1,1,1).to(device) / \
          ((torch.sqrt(1 - scheduler.alpha[t_0].view(batch_size,1,1,1)).to(device)) *
           (torch.sqrt(1 - scheduler.beta[t_0].view(batch_size,1,1,1)).to(device)))

    if REMind_mode == 'SG1':
        x_generated = (1 / (torch.sqrt(1 - scheduler.beta[t_0].view(batch_size,1,1,1)).to(device))) * z - \
                      (temp * model(z, t_0, lr_up=kwargs['lr_up']))
    elif REMind_mode == 'SG2':
        x_generated = (1 / (torch.sqrt(1 - scheduler.beta[t_0].view(batch_size,1,1,1)).to(device))) * z - \
                      (temp * model(z, t_0, c_idx=kwargs['c_idx'], top=kwargs['top'], bottom=kwargs['bottom']))

    return x_generated



def save_inference_img(infenrece_img_path, bidx, img_stack):

    for i in range(img_stack.shape[0]):

        os.makedirs(infenrece_img_path + f'/{bidx}_{i}', exist_ok=True)
        img_path = infenrece_img_path + f'/{bidx}_{i}/{bidx}_sample_{i}.tif'
        tiff.imwrite(img_path,
        img_stack[i,:].numpy(), 
        imagej=True, 
        bigtiff=False) 

def monitoring_SG1_training(test_dataset_path, monitor_process_path, img_size, sf, model, scheduler, num_time_steps, epoch_idx, N_mean, REMind_mode):

        test_dataset = Stage1_Dataset(test_dataset_path,sf=sf, img_size=img_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
        with torch.no_grad():
            model = model.eval()
            for bidx, (x_lr, x_hr) in enumerate(test_loader):
                x_lr = x_lr.cuda()
                x_hr = x_hr.cuda()

                x_lr_up = F.interpolate(x_lr, size=(img_size, img_size), mode='bicubic', align_corners=False)
                x_lr_up = x_lr_up.repeat(N_mean, 1, 1, 1)


                print(f'Monitoring SG1 super-resolution image {bidx}...')                
                img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, N_mean, REMind_mode, lr_up=x_lr_up)
                generated_hr = img_generated[:, 0,:,:]*255

                img_stack_total = torch.mode(generated_hr, dim=0).values
                img_stack_total = img_stack_total *255
                


                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                plt.subplots_adjust(wspace=0.1, hspace=0.1) 
                plt.tight_layout()

                axes[0].imshow(x_hr.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') 
                axes[0].set_title("real HR")
                axes[1].imshow(x_lr.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') 
                axes[1].set_title("LR")
                axes[2].imshow(img_stack_total.cpu().numpy()*255, cmap='viridis') 
                axes[2].set_title("Generated HR")

        
                plt.savefig(monitor_process_path + '/' + f'SR_img_epoch{epoch_idx}_sample_{bidx}.png', bbox_inches='tight', pad_inches=0)
                tiff.imwrite(monitor_process_path + '/' + f'SR_img_epoch{epoch_idx}_sample_{bidx}.tif',img_stack_total.cpu().numpy(), imagej=True, bigtiff=False)           
        model = model.train()

def inference_SG1(model,
                  img_size, 
                  sf, 
                  num_time_steps,  
                  num_epochs, 
                  ema_decay, 
                  lr, 
                  check_point_SG1_inference_path,  
                  test_dataset_path_SG1, 
                  infenrece_img_SG1_path, 
                  batch_size_inference,
                  N_mean,
                  REMind_mode):

    test_dataset = Stage1_Dataset(test_dataset_path_SG1,sf=sf, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_inference, shuffle=False, drop_last=False, num_workers=0)


    checkpoint = torch.load(check_point_SG1_inference_path)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    with torch.no_grad():
        model = ema.module.eval()
        model = model.cuda()
        for bidx, (x_lr, x_hr) in enumerate(tqdm(test_loader)):
            x_lr = x_lr.cuda()
            x_hr = x_hr.cuda()
            x_lr_up = F.interpolate(x_lr, size=(img_size, img_size), mode='bicubic', align_corners=False)
            if batch_size_inference ==1:
                print(f'generating {N_mean} times for a single sample and average--batch_size_inference={batch_size_inference}')
                x_lr_up = x_lr_up.repeat(N_mean, 1, 1, 1)
                img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, N_mean, REMind_mode, lr_up=x_lr_up)
            else:
                img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, batch_size_inference, REMind_mode, lr_up=x_lr_up)
            
            generated_hr = img_generated[:, 0,:,:]*255
            save_inference_img(infenrece_img_SG1_path, bidx, generated_hr.cpu())  

            if True:
                img_stack_total = torch.mode(generated_hr, dim=0).values
                img_stack_total = img_stack_total
                print(f'img_stack_total.shape:{img_stack_total.shape}')
                fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                plt.subplots_adjust(wspace=0.1, hspace=0.1) 
                plt.tight_layout()

                axes[0].imshow(x_hr.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') 
                axes[0].set_title("real HR")
                axes[1].imshow(x_lr.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') 
                axes[1].set_title("LR")
                axes[2].imshow(img_stack_total.cpu().numpy(), cmap='viridis') 
                axes[2].set_title("Generated HR")

def monitoring_SG2_training(test_dataset_path, monitor_process_path, img_size, L, model, scheduler, num_time_steps, epoch_idx, N_mean, REMind_mode):
    test_dataset = SG2_3D_StackDataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
    with torch.no_grad():
        model = model.eval()
        for bidx, (x,_) in enumerate(test_loader):
            x = x.cuda()
            top = x[:, 0, :, :].unsqueeze(dim=1)
            bottom = x[:, L-1, :, :].unsqueeze(dim=1) #1 channel as condition 2
            top = top.repeat(N_mean, 1, 1, 1)
            bottom = bottom.repeat(N_mean, 1, 1, 1)
            img_stack = torch.zeros(N_mean, L, img_size, img_size)
        
            for c in tqdm(range(1,L-1)):
                print(f'generating image {bidx} cahnnel {c}')   
                c_idx = torch.full((N_mean,), c)
                img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, N_mean, REMind_mode, c_idx=c_idx, top=top, bottom=bottom)
                img_stack[:, c,:,:] = img_generated[:, 0,:,:]*255 

            #img_stack_total = torch.mean(img_stack, dim=0)
            img_stack_total = torch.mode(img_stack, dim=0).values
            print(f'img_stack_total.shape:{img_stack_total.shape}')
            img_stack_total[0, :,:] = x[0, 0,:,:]*255
            img_stack_total[L-1,:,:] = x[0, L-1,:,:]*255

            fig, axes = plt.subplots(2, L, figsize=(15, 5))
            plt.subplots_adjust(wspace=0.1, hspace=0.1) 
            plt.tight_layout()
            for c_i in range(L):
                if c_i == 0:
                    axes[0, 0].imshow(x.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') #top real
                    axes[0, 0].set_title("real top")
                    axes[1, 0].imshow(x.cpu().numpy()[0, 0,:,:]*255, cmap='viridis') #top generated (==real)
                    axes[1, 0].set_title("real top")

                elif c_i == L -1:          
                    axes[0, L - 1].imshow(x.cpu().numpy()[0, L-1,:,:]*255, cmap='viridis') #top real
                    axes[0, L - 1].set_title("real bottom")
                    axes[1, L - 1].imshow(x.cpu().numpy()[0, L-1,:,:]*255, cmap='viridis') #top generated (==real)
                    axes[1, L - 1].set_title("real bottom")
                else:
                    axes[0, c_i].imshow(x.cpu().numpy()[0, c_i,:,:]*255, cmap='viridis') #channel 1 real
                    axes[0, c_i].set_title(f"R C{c_i}")
                    axes[1, c_i].imshow(img_stack_total[c_i,:,:]*255, cmap='viridis') #channel 2 generated
                    axes[1, c_i].set_title(f"G C{c_i}")
        
            plt.savefig(monitor_process_path + '/' + f'SG2_img_epoch{epoch_idx}_sample_{bidx}.png', bbox_inches='tight', pad_inches=0)
            tiff.imwrite(monitor_process_path + '/' + f'SG2_img_epoch{epoch_idx}_sample_{bidx}.tif',img_stack_total.cpu().numpy(), imagej=True, bigtiff=False)
    model = model.train()

def Apply_SG1_for_3D_thicker_sample(model,
                    L,
                  img_size, 
                  sf, 
                  num_time_steps,  
                  num_epochs, 
                  ema_decay, 
                  check_point_SG1_inference_path,  
                  sparse_FIB_SEM_low_res_stack_path, 
                  sparse_FIB_SEM_high_res_stack_path, 
                  N_mean,
                  batch_size,
                  REMind_mode):
    test_dataset = LHR_3D_Dataset(sparse_FIB_SEM_low_res_stack_path,sf=sf, img_size=img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    checkpoint = torch.load(check_point_SG1_inference_path)

    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    with torch.no_grad():
        model = ema.module.eval()
        model = model.cuda()
        for bidx, (x_lr, x_hr) in enumerate(tqdm(test_loader)):
            x_lr = x_lr.cuda()
            x_hr = x_hr.cuda()
            generated_hr = torch.zeros(x_hr.shape[0], x_hr.shape[2],  x_hr.shape[3], x_hr.shape[4],).cuda()

            for i in range(x_hr.shape[2]): # loop over the thickness 120x128x128
                if i% (L-1) == 0: # depending on the parameter in SG2 L, e.g., L=4, 6 etc.,
                    x_lr_up = F.interpolate(x_lr[:,:,i, :,:], size=(img_size, img_size), mode='bicubic', align_corners=False)
                    img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, batch_size, REMind_mode, lr_up=x_lr_up)
                    generated_hr[:, i,:,:] = img_generated[:, 0,:,:]*255

            for img_i in range(generated_hr.shape[0]):
                img_path = sparse_FIB_SEM_high_res_stack_path + '/' +f'/{img_i}_sample_{bidx}.tif'
                tiff.imwrite(img_path,
                            generated_hr[img_i,:].cpu().numpy(), 
                            imagej=True, 
                            bigtiff=False) 


def inference_SG2(model,
                  img_size, 
                  num_time_steps,  
                  num_epochs, 
                  ema_decay, 
                  check_point_SG2_inference_path,  
                  test_dataset_path, 
                  infenrece_img_SG2_path, 
                  batch_size_inference,
                  L,
                  N_mean,
                  REMind_mode):
    test_dataset = SG2_3D_StackDataset(test_dataset_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_inference, shuffle=False, drop_last=True, num_workers=0)
    
    checkpoint = torch.load(check_point_SG2_inference_path)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    with torch.no_grad():
        model = ema.module.eval()
        model = model.cuda()
        for bidx, (x,_) in enumerate(test_loader): 
            x = x.cuda()
            top = x[:, 0, :, :].unsqueeze(dim=1) 
            bottom = x[:, L-1, :, :].unsqueeze(dim=1) 

            if batch_size_inference ==1:
                print(f'generating {N_mean} times for a single sample and average--batch_size={batch_size_inference}')
                top = top.repeat(N_mean, 1, 1, 1)
                print(f'top.shape: {top.shape}')
                bottom = bottom.repeat(N_mean, 1, 1, 1)              
                img_stack = torch.zeros(N_mean, L, img_size, img_size)
            else:
                img_stack = torch.zeros(batch_size_inference, L, img_size, img_size)

            for c in tqdm(range(1,L-1)):
                print(f'generating image {bidx} cahnnel {c}') 
                if batch_size_inference ==1:             
                    c_idx = torch.full((N_mean,), c)
                    img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, N_mean, REMind_mode, c_idx=c_idx, top=top, bottom=bottom)
                else:
                    c_idx = torch.full((batch_size_inference,), c)
                    img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, batch_size_inference, REMind_mode, c_idx=c_idx, top=top, bottom=bottom)
                
                img_stack[:, c,:,:] = img_generated[:, 0,:,:]
            
            
            img_stack[:, 0, :,:] = top[:, 0, :, :].cpu()
            img_stack[:, L-1, :,:] = bottom[:, 0, :,:].cpu()
            img_stack_total = torch.mode(img_stack, dim=0).values
            img_stack = img_stack*255
            x = x*255

            fig, axes = plt.subplots(2, L, figsize=(15, 5))
            plt.subplots_adjust(wspace=0.1, hspace=0.1) 
            plt.tight_layout()

            for c_i_inf in range(L):
                axes[0, c_i_inf].imshow(x.cpu().numpy()[0, c_i_inf,:,:], cmap='viridis') 
                axes[0, c_i_inf].set_title("ground truth")

                axes[1, c_i_inf].imshow(img_stack_total[c_i_inf,:,:], cmap='viridis') 
                axes[1, c_i_inf].set_title("DDPM prediction")
            #plt.show()

            save_inference_img(infenrece_img_SG2_path, bidx, img_stack)  


def sparse_to_refined_3D_thicker_sample_SG1_to_SG2(model, L, img_size, num_time_steps, num_epochs, ema_decay, 
                                                check_point_SG2_inference_path,  
                                                sparse_FIB_SEM_high_res_stack_path, 
                                                refined_FIB_SEM_high_res_stack_path,
                                                N_mean,
                                                batch_size,
                                                REMind_mode):
    checkpoint = torch.load(check_point_SG2_inference_path)
    model.load_state_dict(checkpoint['weights'])
    ema = ModelEmaV3(model, decay=ema_decay)
    ema.load_state_dict(checkpoint['ema'])
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)

    test_dataset = SG2_3D_StackDataset(sparse_FIB_SEM_high_res_stack_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

    with torch.no_grad():
        model = ema.module.eval()
        model = model.cuda()
        count=0
        for bidx, (x,_) in enumerate(test_loader):
            count += 1
            x = x.cuda()
            x_org = x.clone()
            N = x.shape[1]//(L-1) #number of chunks for a single thicker sample
            x_input = torch.zeros(batch_size*N, L, img_size, img_size).cuda()

            top = torch.zeros(x_input.shape[0], 1, img_size, img_size).cuda()
            bottom = torch.zeros(x_input.shape[0], 1, img_size, img_size).cuda()
            # Step 1: Split along dim=1 explicitly
            for i in range(batch_size):
                for j in range(N):
                    print(f'{x[i, j*(L-1):(j+1)*(L-1)+1, :, :].shape}')
                    x_input[i*N+j,:,:,:] = x[i, j*(L-1):(j+1)*(L-1)+1, :, :]

            img_stack = torch.zeros(x_input.shape[0], L, img_size, img_size)       
            img_stack_recovered = torch.zeros(batch_size,x.shape[1], img_size, img_size)
            top = x_input[:, 0, :,:].unsqueeze(dim=1) 
            bottom = x_input[:, L-1, :,:].unsqueeze(dim=1) 

            for c in tqdm(range(1,L-1)):
                print(f'generating image {bidx} cahnnel {c}') 

                c_idx = torch.full((batch_size*N,), c)
                img_generated = reverse_sampling(model, scheduler, num_time_steps, img_size, batch_size*N, REMind_mode, c_idx=c_idx, top=top, bottom=bottom)
                img_stack[:, c,:,:] = img_generated[:, 0,:,:]

            img_stack[:, 0, :,:] = x_input[:, 0, :, :].cpu()
            img_stack[:, L-1, :,:] = x_input[:, L-1, :,:].cpu()

            block_id = 0
            for b_i in range(batch_size):
                for n_i in range(N):

                    block_id = b_i*N + n_i

                    img_stack_recovered[b_i, (L-1)*n_i:((n_i+1)*(L-1)+1),:, :] = img_stack[block_id, :, :, :]

            img_stack_recovered = img_stack_recovered*250

            for i in range(img_stack_recovered.shape[0]):
                img_path = refined_FIB_SEM_high_res_stack_path + f'/refined_3D_FIB_SEM_sample_{bidx}_{count}_{i}.tif'
                tiff.imwrite(img_path,
                img_stack_recovered[i,:].cpu().numpy(), 
                imagej=True, 
                bigtiff=False) 