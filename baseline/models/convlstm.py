import torch

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(ConvBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channels = out_channels

        in_channels = [in_channel] + out_channels[:-1]

        for i, (in_channel, out_channel) in enumerate(zip(in_channels, out_channels)):
            stride = 2 if i == 0 else 1
            setattr(self, f'conv_{i}', nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1))
            setattr(self, f'batchnorm_{i}', nn.BatchNorm2d(out_channel))

        self.conv_res = nn.Conv2d(in_channels[0], out_channels[-1], kernel_size=3, stride=2, padding=1)
        self.batchnorm_res = nn.BatchNorm2d(out_channels[-1])
        self.relu = nn.ReLU()
        # compute the output size based on the model architecture

    def forward(self, x):

        x_res = self.conv_res(x)
        x_res = self.batchnorm_res(x_res)

        for i in range(len(self.out_channels)):
            x = getattr(self, f'conv_{i}')(x)
            x = getattr(self, f'batchnorm_{i}')(x)
            x = self.relu(x)

        x = x + x_res
        x = self.relu(x)
        return x

class ConvolutionalModule(nn.Module):
    def __init__(self, in_channel=1, in_conv_channel=64,  
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
        - in_conv_channel (int): Number of output channels for the first convolution
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

        self.input_conv = nn.Conv2d(in_channel, in_conv_channel, kernel_size=7, stride=1, padding=1)
        # construct the resnet blocks
        for i, out_channel in enumerate(out_channels):
            in_ch = in_conv_channel if i == 0 else out_channels[i-1][-1]
            setattr(self, f'resnet_{i}', ConvBlock(in_ch, out_channel))
    
    def forward(self, x):

        x = self.input_conv(x)
        for i in range(len(self.out_channels)):
            x = getattr(self, f'resnet_{i}')(x)

        return x

class ConvLSTM(nn.Module):
    __acceptable_params__ = ['emb_dim', 'hidden_size', 'num_layers', 'num_classes', 'dropout' ]
    def __init__(self, **kwargs):
        super(ConvLSTM, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params__]

        self.conv_module = ConvolutionalModule()
        self.FCN = nn.Linear(16, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.fc1 = nn.Linear(self.hidden_size, 2)
        
    def forward(self, x, tasks=['t1', 't2']):

        x = x.unsqueeze(1)
        x = self.conv_module(x)

        b_size, _, _, seq_len = x.shape
        x = x.view(b_size, seq_len, -1)
        x = self.FCN(x)

        # Forward pass through LSTM
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        t1_out, t2_out = None, None
        if 't1' in tasks:
            t1_out = self.fc1(out[:,-1,:])
        if 't2' in tasks:
            t2_out = self.fc2(out[:,-1,:])
        
        return t1_out, t2_out

if __name__ == '__main__':

    kwargs = {
        'emb_dim': 64,
        'hidden_size': 64,
        'num_layers': 1,
        'num_classes': 6
    }
    model = ConvLSTM(**kwargs)
    print(model)
    x = torch.randn(32, 40, 301)
    t1_out, t2_out = model(x)