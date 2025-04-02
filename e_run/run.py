import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import os
from b_Model_training.Model_training import train_model
from e_run import Parameters


if __name__ == '__main__':

    material_type =Parameters.material_type
    REMind_mode = Parameters.REMind_mode 
    img_size = Parameters.img_size 
    sf = Parameters.sf
    lr =  Parameters.lr
    ema_decay = Parameters.ema_decay
    num_epochs = Parameters.num_epochs
    load_pretrained = Parameters.load_pretrained
    load_pretrained_epoch = Parameters.load_pretrained_epoch
    N_mean = Parameters.N_mean
    input_channels = Parameters.input_channels
    L =Parameters.L

    #paths
    train_dataset_path_SG1 = Parameters.train_dataset_path_SG1
    test_dataset_path_SG1  = Parameters.test_dataset_path_SG1
    train_dataset_path_SG2 = Parameters.train_dataset_path_SG2
    test_dataset_path_SG2 = Parameters.test_dataset_path_SG2
    monitor_process_path = Parameters.monitor_process_path
    checkpoint_path_SG1 = Parameters.checkpoint_path_SG1
    checkpoint_path_SG2 = Parameters.checkpoint_path_SG2

    os.makedirs(checkpoint_path_SG1, exist_ok=True)
    os.makedirs(checkpoint_path_SG2, exist_ok=True)

    #hyperparameters
    num_time_steps = Parameters.num_time_steps
    batch_size = Parameters.batch_size
    num_groups = Parameters.num_groups

    if img_size == 128:
        Channels = [64, 128, 256, 512, 512, 384]
        Attentions = [False, False, False, True, False, False]
        Upscales = [False, False, False, True, True, True]
        
    elif img_size == 256:
        Channels = [32, 64, 128, 256, 512, 512, 384, 256]
        Attentions = [False, False, False, False, True, False, False, False]
        Upscales = [False, False, False, False, True, True, True, True]        
    
    #Train models
    if True:    #Train models
        train_model(Channels=Channels, 
                Attentions=Attentions, 
                Upscales=Upscales, 
                num_groups=num_groups, 
                batch_size=batch_size, 
                img_size=img_size, 
                input_channels = input_channels,
                sf=sf, 
                num_time_steps=num_time_steps,  
                num_epochs=num_epochs, 
                ema_decay=ema_decay, 
                lr=lr, 
                checkpoint_path_SG1=checkpoint_path_SG1, 
                checkpoint_path_SG2=checkpoint_path_SG2, 
                load_pretrained=load_pretrained, 
                load_pretrained_epoch=load_pretrained_epoch, 
                train_dataset_path_SG1=train_dataset_path_SG1,
                test_dataset_path_SG1=test_dataset_path_SG1,
                train_dataset_path_SG2=train_dataset_path_SG2,
                test_dataset_path_SG2=test_dataset_path_SG2,
                monitor_process_path=monitor_process_path,
                N_mean=N_mean,
                REMind_mode=REMind_mode,
                L=L)

