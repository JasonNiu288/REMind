import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
import os
from c_utilities.utilities import inference_SG1, Apply_SG1_for_3D_thicker_sample, inference_SG2,sparse_to_refined_3D_thicker_sample_SG1_to_SG2
from e_run import Parameters
from a_Networks.Networks import UNET


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
    L= Parameters.L
    
    #paths
    test_dataset_path_SG1  = Parameters.test_dataset_path_SG1
    test_dataset_path_SG2  = Parameters.test_dataset_path_SG2

    sparse_FIB_SEM_low_res_stack_path = Parameters.sparse_FIB_SEM_low_res_stack_path

    #inference
    check_point_SG1_inference_path = Parameters.check_point_SG1_inference_path
    check_point_SG2_inference_path = Parameters.check_point_SG2_inference_path
    infenrece_img_SG1_path = Parameters.infenrece_img_SG1_path
    infenrece_img_SG2_path = Parameters.infenrece_img_SG2_path
    sparse_FIB_SEM_high_res_stack_path = Parameters.sparse_FIB_SEM_high_res_stack_path
    refined_FIB_SEM_high_res_stack_path = Parameters.refined_FIB_SEM_high_res_stack_path

    os.makedirs(infenrece_img_SG1_path, exist_ok=True)
    os.makedirs(infenrece_img_SG2_path, exist_ok=True)
    os.makedirs(sparse_FIB_SEM_high_res_stack_path, exist_ok=True)
    os.makedirs(refined_FIB_SEM_high_res_stack_path, exist_ok=True)

    #hyperparameters
    num_time_steps = Parameters.num_time_steps
    batch_size = Parameters.batch_size
    batch_size_inference = Parameters.batch_size_inference
    num_groups = Parameters.num_groups

    if img_size == 128:
        Channels = [64, 128, 256, 512, 512, 384]
        Attentions = [False, False, False, True, False, False]
        Upscales = [False, False, False, True, True, True]
        
    elif img_size == 256:
        Channels = [32, 64, 128, 256, 512, 512, 384, 256]
        Attentions = [False, False, False, False, True, False, False, False]
        Upscales = [False, False, False, False, True, True, True, True]        
    
    #Inference models
    if REMind_mode =='SG1':
        model = UNET(
            Channels = Channels, 
            Attentions = Attentions, 
            Upscales = Upscales,
            num_groups = num_groups,
            dropout_prob = 0.1,
            num_heads = 8,
            input_channels = input_channels, 
            output_channels = 1,
            time_steps = num_time_steps,
            img_size =img_size,
            REMind_mode = REMind_mode,
            ).cuda()
        if False:    #Inference/test the model, Flag it as True if you test the model performance over a series of paris of 2D slices
            inference_SG1(model = model,
                  img_size=img_size, 
                  sf=sf, 
                  num_time_steps=num_time_steps,  
                  num_epochs=num_epochs, 
                  ema_decay=ema_decay, 
                  lr=lr, 
                  check_point_SG1_inference_path=check_point_SG1_inference_path,  
                  test_dataset_path_SG1=test_dataset_path_SG1, 
                  infenrece_img_SG1_path=infenrece_img_SG1_path, 
                  batch_size_inference = batch_size_inference,
                  N_mean=N_mean,
                  REMind_mode=REMind_mode)
        if True:    #Flag it Ture if you apply SG1 model to convert low-res sparse stack into prepare sparse high-resolution 3D stack which will be processed by SG2 model.
            Apply_SG1_for_3D_thicker_sample(model=model,
                                                L=L,
                                            img_size=img_size, 
                                            sf=sf, 
                                            num_time_steps=num_time_steps,  
                                            num_epochs=num_epochs, 
                                            ema_decay=ema_decay, 
                                            check_point_SG1_inference_path=check_point_SG1_inference_path,  
                                            sparse_FIB_SEM_low_res_stack_path=sparse_FIB_SEM_low_res_stack_path, 
                                            sparse_FIB_SEM_high_res_stack_path=sparse_FIB_SEM_high_res_stack_path, 
                                            N_mean=N_mean,
                                            batch_size=1,
                                            REMind_mode=REMind_mode)
    if REMind_mode =='SG2':
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

        if False:
            inference_SG2(model=model,
                        img_size=img_size, 
                        num_time_steps=num_time_steps,  
                        num_epochs=num_epochs, 
                        ema_decay=ema_decay, 
                        check_point_SG2_inference_path=check_point_SG2_inference_path,  
                        test_dataset_path=test_dataset_path_SG2, 
                        infenrece_img_SG2_path=infenrece_img_SG2_path, 
                        batch_size_inference=batch_size_inference,
                        L=L,
                        N_mean=N_mean,
                        REMind_mode=REMind_mode)
        if True:
            sparse_to_refined_3D_thicker_sample_SG1_to_SG2(model=model, L=L, img_size=img_size, num_time_steps=num_time_steps,
                                                           num_epochs=num_epochs, ema_decay=ema_decay, 
                                                           check_point_SG2_inference_path=check_point_SG2_inference_path,  
                                                            sparse_FIB_SEM_high_res_stack_path=sparse_FIB_SEM_high_res_stack_path, 
                                                            refined_FIB_SEM_high_res_stack_path=refined_FIB_SEM_high_res_stack_path,
                                                            N_mean=N_mean, batch_size=1,
                                                            REMind_mode=REMind_mode)