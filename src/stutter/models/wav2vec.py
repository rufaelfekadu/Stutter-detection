
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, AutoFeatureExtractor
import torch
import torch.nn as nn

class Wav2Vec2Classifier(nn.Module):
    def __init__(self, num_classes: int, feature_size: int = 1024):
        super().__init__()
        self.fc = nn.Linear(feature_size, num_classes)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder
    def forward(self, x):
        return self.fc(x)