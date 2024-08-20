import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class YOHO(nn.Module):
    def __init__(self):
        super(YOHO, self).__init__()
        
        # Initial input conv layer
        self.layer1_conv = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        self.layer1_bn = nn.BatchNorm2d(32, eps=1e-4, affine=True)
        
        # Depthwise separable conv layers defined from LAYER_DEFS
        self.layers = nn.ModuleList()
        
        LAYER_DEFS = [
            # ([3, 3], 1,   64),
            # ([3, 3], 2,  128),
            # ([3, 3], 1,  128),
            # ([3, 3], 2,  256),
            # ([3, 3], 1,  256),
            ([3, 3], 2,  512),
            ([3, 3], 1,  512),
            ([3, 3], 1,  512),
            ([3, 3], 1,  512),
            ([3, 3], 1,  512),
            ([3, 3], 1,  512),
            ([3, 3], 2, 1024),
            ([3, 3], 1, 1024),
            ([3, 3], 1,  512),
            ([3, 3], 1,  256),
            ([3, 3], 1,  128),
        ]
        
        # Add layers based on LAYER_DEFS
        in_channels = 32
        for i, (kernel, stride, num_filters) in enumerate(LAYER_DEFS):
            # Depthwise Convolution
            depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel, stride=stride, padding=1, groups=in_channels, bias=False)
            depthwise_bn = nn.BatchNorm2d(in_channels, eps=1e-4, affine=True)
            depthwise_relu = nn.ReLU()

            # Pointwise Convolution
            pointwise = nn.Conv2d(in_channels, num_filters, kernel_size=(1, 1), stride=1, padding=0, bias=False)
            pointwise_bn = nn.BatchNorm2d(num_filters, eps=1e-4, affine=True)
            pointwise_relu = nn.ReLU()

            # Add to module list
            self.layers.append(nn.Sequential(depthwise, depthwise_bn, depthwise_relu, pointwise, pointwise_bn, pointwise_relu))
            in_channels = num_filters
        
        # Final Conv1D for prediction
        self.final_conv1d = nn.Conv1d(in_channels=256, out_channels=6, kernel_size=1)
        
    def forward(self, x):
        # Initial input conv layer
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = F.relu(x)
        
        # Apply depthwise and pointwise convolutions
        for layer in self.layers:
            x = layer(x)
        
        # Reshape for Conv1D
        batch_size, num_channels, height, width = x.size()
        x = x.view(batch_size, num_channels * width, height )  # reshape for Conv1D

        # Final Conv1D for prediction
        x = self.final_conv1d(x)
        x = torch.sigmoid(x)
        
        return x

# Example instantiation and forward pass
# model = YOHO()
# example_input = torch.randn(5, 1, 801, 64)  # Example input: batch size 1, 1 channel, 801x64
# output = model(example_input)
# print(output.shape)