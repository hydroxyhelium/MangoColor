import numpy
import torch 
import torch.nn.init as init
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):

        ##input shape is expected to be (batch_size,6, 512, 512)
        self.down1 = self.downsample(6, 64, 4, False) ## (batch_size, 64, 256, 256)
        self.down2 = self.downsample(64, 128, 4) ## (batch_size, 128, 128, 128)
        self.down3 = self.downsample(128, 256, 4) ## (batch_size, 256, 64, 64)
        self.down4 = self.downsample(256, 512, 4) ## (batch_size, 512, 32, 32)

        self.zero_padding_1 = torch.nn.ZeroPad2d(1) ## (batch_size, 512, 34, 34)
        self.conv = torch.nn.Conv2d(512, 1024, stride=1, bias=False) ##(batch_size, 1024, 31, 31)
        self.custom_initializer(self.conv)
        self.bathnorm_layer = torch.nn.BatchNorm2d(1024) 
        self.leaky_relu = torch.nn.LeakyRelu()
        self.zero_padding_2 = torch.nn.ZeroPad2d(1) ## (batch_size, 1024, 32, 32)
        self.last = torch.nn.Conv2d(1024, 1, stride=1)
        self.custom_initializer(self.last)

    def custom_initializer(self, tensor):
        init._no_grad_normal_(tensor.weight, mean=0.0, std=0.02)
    
    def downsample(self, input_chanels,filters, size, apply_batchnorm=True):
        """ 
        returns sequential layer and the number of channels in it
        """
        layer = torch.nn.Conv2d(input_chanels, filters, size, stride=2, bias=False, padding='same')
        self.custom_initializer(layer)
        batch_norm = None

        seq = None

        if(apply_batchnorm):
            batch_norm = torch.nn.BatchNorm2d(filters)
            seq = torch.nn.Sequential(layer, batch_norm)
            return seq
        
        seq = torch.nn.Sequential(layer)
        return seq
    
    def upsample(self, input_chanels, filters, size, apply_dropout=False):
        
        layer = torch.nn.ConvTranspose2d(input_chanels, filters, size, stride=2, bias=False, padding='same')
        self.custom_initializer(layer)
        batch_norm = torch.nn.BatchNorm2d(filters)
        
        if apply_dropout:
            dropout_layer = torch.nn.Dropout(p=0.5)
            seq = torch.nn.Sequential(layer, batch_norm, dropout_layer)
            return seq
        
        seq = torch.nn.Sequential(layer, batch_norm)

        return seq
    
    def forward(self, input):
        input = self.down1(input)
        input = self.down2(input)
        input = self.down3(input)
        input = self.down4(input)
        input = self.zero_padding_1(input)
        input = self.conv(input)
        input = self.bathnorm_layer(input)
        input = self.leaky_relu(input)
        input = self.zero_padding_2(input)
        input = self.last(input)

        return input 