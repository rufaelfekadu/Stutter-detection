import torch

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channels = out_channels

        in_channels = [in_channel] + out_channels[:-1]

        for i, in_channel, out_channel in enumerate(zip(in_channels, out_channels)):
            setattr(self, f'conv1_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
            setattr(self, f'batchnorm_{i}', nn.BatchNorm2d(out_channel))

        self.conv_res = nn.Conv2d(in_channels[0], out_channels[-1], kernel_size=3, stride=1, padding=1)
        self.batchnorm_res = nn.BatchNorm2d(out_channels[-1])
        self.relu = nn.ReLU()
                               
    def forward(self, x):

        x_res = self.conv_res(x)
        x_res = self.batchnorm_res(x_res)

        for i in range(len(self.out_channels)):
            x = getattr(self, f'conv1_{i}')(x)
            x = getattr(self, f'batchnorm_{i}')(x)
            x = self.relu(x)

        x = x + x_res
        x = self.relu(x)
        return x

class ConvolutionalModule(nn.Module):
    def __init__(self, in_channel=1,  
                 out_channels=[[32, 64, 64],
                                 [64, 128, 128],
                                 [128, 128, 128],
                                 [128, 64, 64],
                                 [64, 64, 32],
                                 [32, 16, 16]], num_blocks=6):
        super(ConvolutionalModule, self).__init__()
        '''
        Convolutional block for the ConvLSTM model as described by https://arxiv.org/pdf/1910.12590

        Parameters:
        - in_channel (int): Number of input channels
        - out_channels (list): List of lists of integers representing the number of output channels for each convolution in the resnet block
                       [Default] [[32, 64, 64],
                                 [64, 128, 128],
                                 [128, 128, 128],
                                 [128, 64, 64],
                                 [64, 64, 32],
                                 [32, 16, 16]]
        - num_block (int): Number of conv blocks
        '''

        self.in_channel = in_channel
        self.out_channels = out_channels
        self.num_blocks = num_blocks

        assert len(out_channels) == num_blocks, 'Number of blocks must match the number of output channels'

        # construct the resnet blocks
        for i, out_channel in enumerate(out_channels):
            in_ch = in_channel if i == 0 else out_channels[i-1][-1]
            setattr(self, f'resnet_{i}', ConvBlock(in_ch, out_channel))
    
    def forward(self, x):

        for i in range(len(self.out_channels)):
            x = getattr(self, f'resnet_{i}')(x)

        return x

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv_module = ConvolutionalModule()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, num_classes)
        self.fc_2 = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # Apply convolutional block
        x = self.input_conv(x)
        
        # Reshape input for LSTM
        batch_size, _, _, _ = x.size()
        x = x.view(batch_size, -1, self.hidden_size)
        
        # Forward pass through LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        # Apply fully connected layer
        out = self.fc_1(out[:, -1, :])
        
        return out