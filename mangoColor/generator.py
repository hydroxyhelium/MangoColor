import numpy
import torch 
import torch.nn.init as init
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        """
        note that the training data will have hieght and width 512, 512 respectively 
        """
        
        self.downsample_layer1 = self.downsample(3, 32,4, apply_batchnorm=False) ## (batch_size, 32, 216, 216)
        self.downsample_layer2 = self.downsample(32, 64, 4) ## (batch_size, 64, 128, 128)
        self.downsample_layer3 = self.downsample(64, 128, 4) ## (batch_size, 128, 64, 64)
        self.downsample_layer4 = self.downsample(128, 256, 4) ## (batch_size, 256, 32, 32)
        self.downsample_layer5 = self.downsample(256, 512, 4) ## (batch_size, 512, 16, 16)
        self.downsample_layer6 = self.downsample(512, 512, 4) ## (batch_size, 512,8,8)
        self.downsample_layer7 = self.downsample(512, 512, 4) ## (batch_size, 512,4, 4)
        self.downsample_layer8 = self.downsample(512, 512, 4) ## (batch_size, 512,2,2)
        self.downsample_layer9 = self.downsample(512, 512, 4) ## (batch_size, 512, 1, 1)

        self.down_stack = [self.downsample_layer1, self.downsample_layer2, self.downsample_layer3, self.downsample_layer4, 
                           self.downsample_layer5, self.downsample_layer6, self.downsample_layer7, self.downsample_layer8, self.downsample_layer9]

        self.upsample_layer1 = self.upsample(512,512, 4, apply_dropout=True) ##(batch_size, 1024, 2, 2)
        self.upsample_layer2 = self.upsample(1024, 512, 4, apply_dropout=True) ##(batch_size, 1024, 4,4)
        self.upsample_layer3 = self.upsample(1024, 512, 4, apply_dropout=True) ##(batch_size, 1024,8,8)
        self.upsample_layer4 = self.upsample(1024, 512, 4) ##(batch_size, 1024,16,16)
        self.upsample_layer5 = self.upsample(1024, 512, 4) ##(batch_size, 1024, 32, 32)
        self.upsample_layer6 = self.upsample(1024, 256, 4) ##(batch_size, 512, 64,64)
        self.upsample_layer7 = self.upsample(512, 128, 4) ##(batch_size, 256, 128,128)
        self.upsample_layer8 = self.upsample(256, 64, 4) ##(batch_size, 128, 256,256)

        self.up_stack = [self.upsample_layer1, self.upsample_layer2, self.upsample_layer3, self.upsample_layer4, self.upsample_layer5, self.upsample_layer6, 
                         self.upsample_layer7, self.upsample_layer8]

        self.final_layer = torch.nn.ConvTranspose2d(128, 3, 4, stride=2, padding='same') ## (batch_size, 3, 512, 512)
        self.tanh_layer = nn.tanh()

        self.custom_initializer(self.final_layer)

        return
    
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
        stack = []
        ## input is (batch_size, 512, 512, 3)
        for layer in self.down_stack:
            input = layer(input)
            stack.append(input)
        
        stack = reversed(stack[:-1])

        for up_layer, prev in zip(self.up_stack, stack):
            input = up_layer(input)
            input = torch.cat((input, prev), dim=1) ## need to concatenate dim=2 #(BATCH_SIZE, CHANNELS, height, wieght) 

        input = self.final_layer(input)

        return self.tanh_layer(input)