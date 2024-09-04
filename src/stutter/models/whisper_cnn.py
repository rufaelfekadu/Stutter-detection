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
from transformers import AutoFeatureExtractor, WhisperModel

class WhisperDetector(nn.Module):
    __acceptable_params = ['name', 'num_classes', 'embed_features']
    def __init__(self, embed_features=512, num_classes=14, **kwargs):
        super(WhisperDetector, self).__init__()
        [setattr(self, k, kwargs.get(k, None)) for k in self.__acceptable_params]
        model = WhisperModel.from_pretrained("openai/whisper-base", cache_dir="./outputs")
        self.encoder = model.encoder
        self.embed_features = embed_features
        self.num_classes = num_classes
        
        LAYER_DEFS = [
            (5, 1, 256),
            (5, 2, 128),
            (5, 2, 128),
            # (5, 2, 128),
            (5, 2, 64),
            (5, 1, 64),
            (5, 2, 36),
            (3, 1, 15),
            # (3, 2, 14),
            
        ]
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_channels=self.embed_features, out_channels=256, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )
        
        self.layers = nn.ModuleList()
        # Add layers based on LAYER_DEFS
        in_channels = 256
        for i, (kernel, stride, num_filters) in enumerate(LAYER_DEFS):
            # Depthwise Convolution
            depthwise = nn.Sequential(
                nn.Conv1d(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel, stride=stride, padding=1),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(),
            )

            self.layers.append(depthwise)
            in_channels = num_filters       
        
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.fc2 = nn.Linear(512, 140)
    
    def forward(self, x):
        # x ->  (batch_size, num_mels, seq_len) (batch_size, 80, 3000)
        with torch.no_grad():
            x = self.encoder(x)['last_hidden_state'] # x -> (batch_size, 1500, 512)
            
        # post lexical feature extraction network.
        x = x.permute(0, 2, 1) # x -> (batch_size, 512, 1500)
        x = self.input_conv(x)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.max_pool(x)
        # permute to (batch_size, seq_len, num_mels)
        x = x.permute(0, 2, 1)
        return x

if __name__ == "__main__":
    model = WhisperDetector()
    example_input = torch.randn(5, 80, 3000)  # Example input: batch size 1, 1 channel, 801x64
    output = model(example_input)
    print(output.shape)
    
        