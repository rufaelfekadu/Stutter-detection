from transformers import Wav2Vec2ForSequenceClassification, VivitForVideoClassification
from transformers import Wav2Vec2Config, VivitConfig
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss



_HIDDEN_STATES_START_POSITION = 2

class AudioExtractor(Wav2Vec2ForSequenceClassification):
  def __init__(self, config):
    super().__init__(config)
    self.freeze_base_model()
  def forward(
      self,
      input_values,
      attention_mask=None,
      output_attentions=None,
      output_hidden_states=None,
      return_dict=None,
  ):
    output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

    outputs = self.wav2vec2(
        input_values,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )
    if self.config.use_weighted_layer_sum:
        hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
        hidden_states = torch.stack(hidden_states, dim=1)
        norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
        hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
    else:
        hidden_states = outputs[0]

    hidden_states = self.projector(hidden_states)
    if attention_mask is None:
        pooled_output = hidden_states.mean(dim=1)
    else:
        padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
        hidden_states[~padding_mask] = 0.0
        pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
    return pooled_output

class VideoExtractor(VivitForVideoClassification):
  def __init__(self, config):
    super().__init__(config)
    self.vivit.requires_grad_(False)
    self.vivit.embeddings.requires_grad_(True)
    self.vivit.encoder.layer[0].requires_grad_(True)
    self.vivit.encoder.layer[11].requires_grad_(True)
    self.classifier.requires_grad_(True)
    
  def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ):
    outputs = self.vivit(
        pixel_values,
        head_mask=head_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        interpolate_pos_encoding=interpolate_pos_encoding,
        return_dict=return_dict,
    )

    sequence_output = outputs[0][:, 0, :]
    return sequence_output

class MultiModalClassification(nn.Module):
    def __init__(self, num_labels, wav2vec2_config, vivit_config):
      super().__init__()
      self.num_labels = num_labels
      self.audio_extractor  = AudioExtractor.from_pretrained("superb/wav2vec2-base-superb-ks",config=wav2vec2_config, cache_dir=CACHE_DIR)
      self.video_extractor = VideoExtractor.from_pretrained("google/vivit-b-16x2-kinetics400", config=vivit_config, ignore_mismatched_sizes=True, cache_dir=CACHE_DIR)
      self.activation = nn.Tanh()
      self.projector = nn.Linear(1024, 4096)
      self.classifier = nn.Linear(4096, self.num_labels)
      
    def forward(self, pixel_values,input_values, attention_mask, labels):
      audio = self.audio_extractor(input_values,attention_mask)
      video = self.video_extractor(pixel_values)
      x = torch.cat((video, audio), dim=1)
      output = self.projector(x)
      logits = self.classifier(self.activation(output))
      
      # loss = None
      # if labels is not None:
      #   loss_fct = CrossEntropyLoss()
      #   loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
      loss = None
      if labels is not None:
          pos_weight = torch.tensor([1349/185, 1384/150, 1481/53, 1225/309, 1134/500,1354/180, 924/610 , 1346/188]).to(labels.device)
          loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
          loss = loss_fct(logits.view(-1, self.num_labels), 
                          labels.float().view(-1, self.num_labels))
        
      return (loss, logits)