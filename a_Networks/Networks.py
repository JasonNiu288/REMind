import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import numpy as np
from einops import rearrange #pip install einops

from c_utilities.utilities import Low_res_img_and_Time_Step_Embeddings,slicesEmbeddings




# Residual Blocks
class ResBlock(nn.Module):
    def __init__(self, C: int, num_groups: int, dropout_prob: float, embed_dim:int):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.conv_skip = nn.Conv2d(C, C, kernel_size=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)

    def forward(self, x):
    
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        x = self.conv_skip(x)

        return r + x

class Attention(nn.Module):
    def __init__(self, C: int, num_heads:int , dropout_prob: float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]  # the input x is assumed be a 4D tensor [batch_size, channel, height, width]
        x = rearrange(x, 'b c h w -> b (h w) c') # rearrange x to have a shape [batch_size, sequence_length(h*w), channel]
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads) # the 'C' in K b H L C equals C_original // num_heads
        q,k,v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q,k,v, is_causal=False, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')

class UnetLayer(nn.Module):
    def __init__(self, 
            upscale: bool, 
            attention: bool, 
            num_groups: int, 
            dropout_prob: float,
            num_heads: int,
            C: int,
            embed_dim: int):
        super().__init__()
        self.ResBlock1 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob, embed_dim = embed_dim)
        self.ResBlock2 = ResBlock(C=C, num_groups=num_groups, dropout_prob=dropout_prob, embed_dim = embed_dim)
        
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, kernel_size=4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, kernel_size=3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads=num_heads, dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.ResBlock1(x)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x)

        return self.conv(x), x    #the second term self.pool(x) was supposed to be x, which is prepared for next U-net layer

class UNET(nn.Module):
    def __init__(self,
            Channels = [64, 128, 256, 512, 512, 384],
            Attentions = [False, True, False, False, False, True],
            Upscales = [False, False, False, True, True, True],
            num_groups = 32,
            dropout_prob: float = 0.1,
            num_heads: int = 8,
            input_channels: int = 3,
            output_channels: int = 1,
            time_steps: int = 1000,
            L: int = 6,
            img_size = 32,
            REMind_mode: str=''
            ):
        super().__init__()
        self.REMind_mode = REMind_mode
        self.embed_dim = int(img_size * img_size)
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], kernel_size=3, padding=1)
        out_channels = (Channels[-1]//2)+Channels[0]
        self.late_conv = nn.Conv2d(out_channels, out_channels//2, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(out_channels//2, output_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        if REMind_mode == 'SG1':
            self.embeddings_total = Low_res_img_and_Time_Step_Embeddings(time_steps=time_steps, img_size = img_size)
        elif REMind_mode =='SG2':
            self.embeddings_total = slicesEmbeddings(time_steps=time_steps, L=L, img_size = img_size)
        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                num_groups = num_groups,
                attention=Attentions[i],
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads,
                embed_dim = self.embed_dim
            )
            setattr(self, f'Layer{i+1}', layer)


    def forward(self, x, t, **kwargs):
        if self.REMind_mode =='SG1':
            embedding_total = self.embeddings_total(x, t, kwargs['lr_up'])
        elif self.REMind_mode =='SG2':
            embedding_total = self.embeddings_total(x, t, kwargs['c_idx'], kwargs['top'], kwargs['bottom'])
        x = torch.concat((x, embedding_total), dim =1)
        x = self.shallow_conv(x)
        residuals = []
        for i in range(self.num_layers//2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x)
            residuals.append(r)
        for i in range(self.num_layers//2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x)[0], residuals[self.num_layers-i-1]), dim=1)
        return self.output_conv(self.relu(self.late_conv(x)))
        
