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
            nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )


        
    def forward(self, x, output_attentions=False):
        # x ->  (batch_size, num_mels, seq_len) (batch_size, 80, 3000)
        with torch.no_grad():
            x = self.encoder(x) # x -> (batch_size, 1500, 512)

        x = self.dense(x)    # output -> (batch_size, 1500, 5)
        return x

if __name__ == "__main__":
    kwargs = {
        'name': 'sednet',
        'num_classes': 5,
        'emb_dim': 768,
        'encoder_name': 'wav2vec2-b',
        'cache_dir': './outputs'
    }
    model = SedNet(**kwargs)
    example_input = torch.randn(5, 40000)  # Example input: batch size 1, 1 channel, 801x64
    output = model(example_input)
    print(output.shape)
    
        