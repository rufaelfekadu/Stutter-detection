"""
Use whisper as stutter detection and multi-label classification model
1. Lexical feature extraction using whisper
2. CONV layers for classification&detection

Label: start_time, end_time, LABEL_X , start_time, end_time, LABEL_X, start_time, end_time, LABEL_X, ..., stop_token, padding
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel, Wav2Vec2ForCTC 

encoder_bank = {
    'whisper-b': 'openai/whisper-base',
    'whisper-l': 'openai/whisper-large-v3',
    'wav2vec2-b': 'facebook/wav2vec2-base-960h',
}

def get_encoder(encoder_name, cache_dir):
    if 'whisper' in encoder_name:
        return WhisperEncoder(encoder_name, cache_dir)
    elif 'wav2vec2' in encoder_name:
        return Wav2Vec2Encoder(encoder_name, cache_dir)
    else:
        raise ValueError(f"Encoder {encoder_name} not supported")

class WhisperEncoder(nn.Module):
    def __init__(self, encoder_name, cache_dir):
        super(WhisperEncoder, self).__init__()
        self.encoder = WhisperModel.from_pretrained(encoder_name, cache_dir=cache_dir).encoder
    def forward(self, x):
        return self.encoder(x)['last_hidden_state']

class Wav2Vec2Encoder(nn.Module):
    def __init__(self, encoder_name, cache_dir):
        super(Wav2Vec2Encoder, self).__init__()
        self.encoder = Wav2Vec2ForCTC.from_pretrained(encoder_name, cache_dir=cache_dir).wav2vec2
    def forward(self, x):
        
        return self.encoder(x)['last_hidden_state']

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.conv1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # If in_features and out_features are different, we need to match dimensions
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out.permute(0,2,1)).permute(0,2,1)
        out = self.relu(out)
        out = F.dropout(out, 0.3)

        out = self.conv2(out)
        out = self.bn2(out.permute(0,2,1)).permute(0,2,1)
        out = self.relu(out)
        out = F.dropout(out, 0.3)

        out += residual  # Add the skip connection
        out = self.relu(out)  # Apply activation after the addition
        
        
        return out

class MLPBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super(MLPBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.fc = nn.Linear(in_features, out_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features) 

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x.permute(0,2,1)).permute(0,2,1)
        x = self.act(x)
        x = self.dropout(x)
        return x
   
class SedNet(nn.Module):
    __acceptable_params = ['name', 'num_classes', 'emb_dim', 'encoder_name', 'cache_dir']
    def __init__(self, num_classes=5, **kwargs):
        super(SedNet, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        
        self.encoder = get_encoder(encoder_bank[self.encoder_name], cache_dir=self.cache_dir)

        # freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.emb_dim
        self.num_classes = num_classes

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1500)
        
        # self.dense = nn.Sequential(
        #     ResidualBlock(self.emb_dim, 512),
        #     ResidualBlock(512, 512),
        #     ResidualBlock(512, 256),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_classes)
        # )
        self.dense = nn.Sequential(
            MLPBlock(self.emb_dim, 512),
            MLPBlock(512, 512),
            MLPBlock(512, 256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.num_classes)
        )


        
    def forward(self, x, output_attentions=False):
        # x ->  (batch_size, num_mels, seq_len) (batch_size, 80, 3000)
        with torch.no_grad():
            x = self.encoder(x) # x -> (batch_size, 1500, 512)

        x = self.dense(x)    # output -> (batch_size, 1500, 5)
        return x
    
class YAMNet(nn.Module):
    __acceptable_params = ['name', 'output_size']
    def __init__(self, **kwargs):
        super(YAMNet, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        # Define the YAMNet layers using MobileNetV1 blocks
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # Depthwise separable convolution blocks
        self.depthwise_separable_conv_blocks = nn.Sequential(
            self._depthwise_separable_conv(32, 64, stride=1),
            self._depthwise_separable_conv(64, 128, stride=2),
            self._depthwise_separable_conv(128, 128, stride=1),
            self._depthwise_separable_conv(128, 256, stride=2),
            self._depthwise_separable_conv(256, 256, stride=1),
            self._depthwise_separable_conv(256, 512, stride=2),
            *[self._depthwise_separable_conv(512, 512, stride=1) for _ in range(5)],
            self._depthwise_separable_conv(512, 1024, stride=2),
            self._depthwise_separable_conv(1024, 1024, stride=1),
        )

        # Final 1x1 convolution to map to class predictions
        self.fc2 = nn.Conv2d(1024, self.output_size, kernel_size=1, stride=1)
        self.fc1 = nn.Conv2d(1024, 1, kernel_size=1, stride=1)


        # Adaptive average pooling to retain temporal dimension and reduce frequency to 1
        self.temporal_pooling = nn.AdaptiveAvgPool2d((None, 1))  # Output shape: [B, C, T, 1]


    def _depthwise_separable_conv(self, in_channels, out_channels, stride):
        """
        Create a depthwise separable convolution block consisting of:
        - Depthwise convolution
        - Batch Normalization
        - ReLU activation
        - Pointwise convolution
        - Batch Normalization
        - ReLU activation
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, tasks=['t1', 't2']):

        # Input shape: [B, 1, T, F] where T is the time dimension and F is the frequency dimension
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # Pass through depthwise separable convolution blocks
        x = self.depthwise_separable_conv_blocks(x)

        # Apply the classifier to retain temporal dimension
        out1, out2 = None, None
        if 't1' in tasks:
            out1 = self.temporal_pooling(self.fc1(x)).squeeze(-1).permute(0, 2, 1)
        if 't2' in tasks:
            out2 = self.temporal_pooling(self.fc2(x)).squeeze(-1).permute(0, 2, 1)

        return out1, out2

class CRNN(nn.Module):
    __acceptable_params = ['output_size', 'hidden_size', 'num_layers']
    def __init__(self, **kwargs):
        super(CRNN, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        
        # Convolutional feature extraction layers
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1),  # [B, 1, T, F] -> [B, 64, T, F]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 5)),  # Pooling over frequency dimension [B, 64, T, F] -> [B, 64, T, F/2]

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),  # [B, 128, T, F/2] -> [B, 128, T, F/4]

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((1, 2)),  # [B, 256, T, F/4] -> [B, 256, T, F/8]
        )

        # GRU recurrent layer to capture temporal dependencies
        self.rnn = nn.LSTM(
            input_size=2*128,   # Input size matches the output channels of the CNN
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True
        )

        # Final classifier layer to output class scores
        self.fc2 = nn.Linear(2*self.hidden_size , self.output_size)  # *2 because bidirectional
        self.fc1 = nn.Linear(2*self.hidden_size, 1)  # 

    def forward(self, x, tasks=['t1', 't2']):
        x = x.unsqueeze(1)
        # Input: x of shape [B, 1, T, F] (batch, channels, time, frequency)
        x = self.conv_block(x)  # Feature extraction through CNN layers

        # Reshape for RNN input: [B, C, T, F] -> [B, T, C*F]
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()  # [B, C, T, F] -> [B, T, C, F]
        x = x.view(B, T, C * F)  # Flatten frequency and channel dims: [B, T, C*F]

        # Pass through the GRU
        rnn_out, _ = self.rnn(x)  # rnn_out shape: [B, T, rnn_hidden_size]

        # Classify each time step: [B, T, 2*rnn_hidden_size] -> [B, T, C]
        output1, output2 = None, None
        if 't1' in tasks:
            output1 = self.fc1(rnn_out)
        if 't2' in tasks:
            output2 = self.fc2(rnn_out)
            

        return output1, output2

if __name__ == "__main__":
    kwargs = {
        'name': 'sednet',
        'output_size': 6,
        'emb_dim': 768,
        'encoder_name': 'wav2vec2-b',
        'cache_dir': './outputs',
        'hidden_size': 32,
        'num_layers': 2
    }
    model = CRNN(**kwargs)
    example_input = torch.randn(5, 1501,40)  # Example input: batch size 1, 1 channel, 801x64
    output = model(example_input)
    print(output[0].shape, output[1].shape)
    
        