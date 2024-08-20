"""
Using whisper as stutter detection model
Labels: <0.0> <Stutter event>  <0.1>  <0.2> <Stutter event> <0.3> <0.6> <Stutter event> <0.7> <0.9> <Stutter event> <1.0> 
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WhisperModel


model = WhisperModel.from_pretrained("openai/whisper-base", cache_dir="/tmp/")


        