# compression and denoise model
#
import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from einops import rearrange
from einops.layers.torch import Rearrange

class AutoEncoder(nn.Module):
    def __init__(self, feature_num, layer_num):
        super(AutoEncoder, self).__init__()
        self.feature_num = feature_num
        self.feature_stride = 2

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(layer_num):
            encoder_layers = nn.Sequential(
                nn.ZeroPad2d(1),
                nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(feature_num, feature_num // self.feature_stride, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_num // self.feature_stride, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
            self.encoder.append(encoder_layers)
            feature_num = feature_num // self.feature_stride

        feature_num = self.feature_num
        for i in range(layer_num):
            decoder_layers = nn.Sequential(
                nn.ConvTranspose2d(feature_num // 2, feature_num, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU(),
                nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, bias=False, padding=1),
                nn.BatchNorm2d(feature_num, eps=1e-3, momentum=0.01),
                nn.ReLU()
            )
            self.decoder.append(decoder_layers)
            feature_num //= 2

    def forward(self, x):
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        for decoder_layer in reversed(self.decoder):
            x = decoder_layer(x)

        return x

def spatial_sampling(agent_feature, compress_ratio: float = 0.8):
    """
    spatial sampling except the ego feature
    """
    agent_num, _, H, W = agent_feature.size()
    reduced_pixels = int(H * W * compress_ratio)

    for i in range(1, agent_num):  # Start from index 1 to skip the ego feature # agent_feature[i] [C,H,W]
        aggregate_features = torch.sum(agent_feature[i], dim=0)  # [H,W]
        aggregate_features = aggregate_features.reshape(-1)
        _, indices = torch.topk(aggregate_features, reduced_pixels,)

        mask = torch.zeros_like(aggregate_features)
        mask[indices] = 1
        mask = mask.reshape(H, W)
        agent_feature[i] *= mask

    return agent_feature



def linear_map(x, range=[-1.0, 1.0]):
    span = range[1] - range[0]
    k = span / (torch.max(x) - torch.min(x))
    b = range[1] - torch.max(x) * k
    return (k * x + b)


class NoiseModel(nn.Module):
    def __init__(self,input_dim:int,data_dict=None,kernel_size=5,c_sigma=1.0):
        super().__init__()
        self.data_dict = data_dict
        self.gaussian_filter = nn.Conv2d(input_dim,input_dim, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.init_gaussian_filter(kernel_size, c_sigma)
        self.gaussian_filter.requires_grad = False

    def init_gaussian_filter(self,k_size=5, sigma=1.0):
        """
        where2comm gaussian filter
        """
        center = k_size // 2
        x, y = np.mgrid[0 - center: k_size - center, 0 - center: k_size - center]
        gaussian_kernel = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    
    def forward(self,agent_feature,mu:float=0,std:float=1):
        N, _, H, W = agent_feature.size() # [N,C,H,W]
        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)
        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)
        distance_tensor = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(N, 1, 1,1)
        distance_tensor = linear_map(distance_tensor, range=[0.0, 1.0])

        agent_feature = self.gaussian_filter(agent_feature[1:])

        agent_feature = agent_feature + distance_tensor*agent_feature 

        return agent_feature


def add_noise(agent_feature,mu: float = 0,std: float = 1):
    # Add Gaussian noise to the input
    N, C, H, W = agent_feature.size()
    gaussian_noise = torch.randn(N-1, C, H, W, device=agent_feature.device) * std + mu
    noisy_x = agent_feature.clone()
    noisy_x[1:] += gaussian_noise
    return noisy_x


def add_loc_noise(agent_feature,mu: float = 0,std: float = 1,data_dict=None):
    # Add Gaussian noise to the input
        N, C, H, W = agent_feature.size() # [N,C,H,W]
        # [N,5,5,4,4] 
        # print(data_dict["spatial_correction_matrix"].shape)
        # print(data_dict["pairwise_t_matrix"].shape)
        lin_h = torch.linspace(0, H - 1, H).cuda()
        lin_w = torch.linspace(0, W - 1, W).cuda()
        y, x = torch.meshgrid(lin_h, lin_w)

        y = torch.abs(y - H / 2 + 0.5) if H % 2 == 1 else torch.abs(y - H / 2)
        x = torch.abs(x - W / 2 + 0.5) if W % 2 == 1 else torch.abs(x - W / 2)

        distance_tensor = torch.sqrt(x ** 2 + y ** 2).unsqueeze(0).unsqueeze(0).repeat(N, 1, 1,
                                                                                    1)
        distance_tensor = linear_map(distance_tensor)
 
        gaussian_noise = torch.randn(N-1, C, H, W, device=agent_feature.device) + distance_tensor[1:] + mu
        
        noisy_x = agent_feature.clone()
        noisy_x[1:] += gaussian_noise
        return noisy_x


class MultiDenoiseModel(nn.Module):
    """
    encode and add noise and decode progressively 
    """
    def __init__(self,input_dim:int,compress_ratio:float=0.8,layer_num:int=3) -> None:
        super().__init__()
        self.feature_num = input_dim
        self.factor = 2
        self.compress_ratio = compress_ratio
        # self.noise_model = NoiseModel()
        self.encoderList = nn.ModuleList()
        self.decoderList = nn.ModuleList()
        for _ in range(layer_num):
            encoder_layers = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(input_dim),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(input_dim),
                nn.Dropout(0.5),
                nn.Conv2d(input_dim, input_dim * self.factor, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(input_dim * self.factor),
                nn.Conv2d(input_dim* self.factor, input_dim* self.factor, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.encoderList.append(encoder_layers)

            input_dim = input_dim * self.factor
            self.noise_model = NoiseModel(input_dim)
            setattr(self, f'noise_model_{_}', self.noise_model)

        for _ in range(layer_num):
            decoder_layers = nn.Sequential(
                nn.Conv2d(input_dim, input_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(input_dim),
                nn.ConvTranspose2d(input_dim , input_dim , kernel_size=2, stride=2),
                nn.BatchNorm2d(input_dim),
                nn.Dropout(0.5),
                nn.Conv2d(input_dim , input_dim // self.factor, kernel_size=3, stride=1,  padding=1),
                nn.BatchNorm2d(input_dim// self.factor),
                nn.Conv2d(input_dim// self.factor, input_dim// self.factor, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.decoderList.append(decoder_layers)
            input_dim //= self.factor

    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x


    def forward(self, spatial_features,record_len,data_dict) -> None :
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            """
            add noise and design a model to remove 
            """
            for idx,encoder in enumerate(self.encoderList):
                # TODO just fusion the neighborhood feature
                batch_spatial_feature = encoder(batch_spatial_feature)
                if self.training:
                    # Encode and decode the noisy input
                    noise_model = getattr(self, f'noise_model_{idx}')
                    batch_spatial_feature = noise_model(batch_spatial_feature)
                    # batch_spatial_feature = self.noise_model(batch_spatial_feature,std=idx)
            encoded_feat = spatial_sampling(batch_spatial_feature,self.compress_ratio)
            for decoder in self.decoderList:
                encoded_feat = decoder(encoded_feat)
            out.append(encoded_feat)
        out = torch.cat(out, dim=0)
        return out
            


class MultiDenoiseModelV2(nn.Module):
    """
    add noise and remove using autoencoder
    """
    def __init__(self, input_dim, layer_num:int=3,compress_ratio:float=0.8,training:bool=True):
        super().__init__()
        self.training = training
        self.compress_ratio = compress_ratio
        self.autoencoderList = nn.ModuleList()
        for _ in range(layer_num):
            self.autoencoderList.append(AutoEncoder(input_dim, layer_num))
   
    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x

    def forward(self,spatial_features,record_len=None): # spatial_features e.g. [5,C,H,W]
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            """
            add noise and design a model to remove 
            """
            for autoencoder in self.autoencoderList:
                if self.training:
                    # Encode and decode the noisy input
                    batch_spatial_feature = add_noise(batch_spatial_feature)
                batch_spatial_feature = spatial_sampling(batch_spatial_feature,self.compress_ratio)
                decoded_feature = autoencoder(batch_spatial_feature)
            out.append(decoded_feature)
        out = torch.cat(out, dim=0)
        return out


class DenoiseModel(nn.Module):
    """
    encode ,noise,and decode 
    """
    def __init__(self, input_dim, compress_ratio:float=0.8,training:bool=True):
        super().__init__()
        self.training = training
        self.compress_ratio = compress_ratio
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim*compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim*compress_ratio, eps=1e-3, momentum=0.01),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(in_features=input_dim*compress_ratio, out_features=input_dim),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(input_dim, input_dim*compress_ratio, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm2d(input_dim),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Conv2d(in_features=input_dim, out_features=input_dim),
            nn.Sigmoid(),
        )

   

    def forward(self,spatial_features,record_len=None): # spatial_features e.g. [5,C,H,W]
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            """
            add noise and design a model to remove 
            """
            encoded = self.encoder(batch_spatial_feature)
            if self.training:
                encoded = add_noise(encoded)
            encoded = spatial_sampling(encoded,self.compress_ratio)
                # Encode and decode the noisy input
            decoded = self.decoder(encoded)
            out.append(decoded)
        out = torch.cat(out, dim=0)
        return out


    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    

class DenoiseModelV2(nn.Module):
    """
    denoise using autoencoder
    """
    def __init__(self, input_dim, layer_num:int=3,compress_ratio:float=0.8,training:bool=True):
        super().__init__()
        self.training = training
        self.compress_ratio = compress_ratio
        self.autoencoder = AutoEncoder(input_dim, layer_num)
   

    def forward(self,spatial_features,record_len=None): # spatial_features e.g. [5,C,H,W]
        assert spatial_features.ndim == 4, "tensor dim not correct"
        split_x = self.regroup(spatial_features, record_len) #spatial_features [5,C,H,W]
        out = []
        for batch_spatial_feature in split_x: # [2,C,H,W]
            """
            add noise and design a model to remove 
            """
            if self.training:
                # Encode and decode the noisy input
                batch_spatial_feature = add_noise(batch_spatial_feature)
            batch_spatial_feature = spatial_sampling(batch_spatial_feature,self.compress_ratio)
            decoded_feature = self.autoencoder(batch_spatial_feature)
            out.append(decoded_feature)
        out = torch.cat(out, dim=0)
        return out


    @staticmethod
    def regroup(x, record_len):
        cum_sum_len = torch.cumsum(record_len, dim=0)
        split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
        return split_x
    