import torch

import torch.nn as nn

class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ConvLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv_block = ConvolutionalBlock(3, 64)  # Example: input has 3 channels
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, num_classes)
        self.fc_2 = nn.Linear(hidden_size, 2)
        
    def forward(self, x):
        # Apply convolutional block
        x = self.conv_block(x)
        
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