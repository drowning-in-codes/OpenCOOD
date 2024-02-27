import torch
import torch.nn as nn

from opencood.models.sub_modules.covisformer import CoVisFormer
from torchsummary import summary
from functools import partial
import numpy as np


total_size = []
compressor = CoVisFormer(num_vehicles=5,feature_height=100,feature_width=352,channels=384,).to("cuda")

for name, param in compressor.named_parameters():
    """
    输出网络各层参数大小
    """
    size = param.element_size()
    ele = param.numel()
    total_size.append(size*ele)
    print(f"{name}层参数大小为{size}B,参数个数为{ele},大小约为{(size*ele)/1024/1024}MB,约为{(size*ele)/1024/1024/1024}GB")
t_size = sum(total_size)
print(f"总大小约为{t_size/1024/1024}MB,约为{t_size/1024/1024/1024}GB")
index = np.argsort(total_size)
for k in range(5):
    print(f"参数层是{list(compressor.named_parameters())[index[k]][0]},第{k+1}大的参数大小约为{total_size[index[k]]/1024/1024}MB,约为{total_size[index[k]]/1024/1024/1024}GB")