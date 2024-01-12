import torch
import torch.nn as nn


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

class DenoiseModel(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self,x):
        pass