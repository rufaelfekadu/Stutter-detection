import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments, AdamW, VivitConfig, VivitForVideoClassification, VivitImageProcessor, AutoModelForSequenceClassification
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from transformers import Wav2Vec2Config
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


device = "cuda"
CACHE_DIR = "/tmp/"

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
 
if __name__ == "__main__":
  df = load_from_disk("/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_train")
  df = df.train_test_split(test_size=0.1, seed=42, shuffle=True)
  dataset = df['train']
  val_dataset = df['test']
  test_dataset =load_from_disk("/fsx/homes/Hawau.Toyin@mbzuai.ac.ae/Stutter-detection/outputs/fluencybank/dataset/stutter_hf/ds_5_multimodal_test")
  dataset = dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
  val_dataset = val_dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
  test_dataset = test_dataset.map(lambda x: {"input_values": torch.tensor(x["input_values"][0]), 'attention_mask':torch.tensor(x['attention_mask'][0]), "labels": torch.tensor(x["labels"])})
  
  dataloader= DataLoader(test_dataset, batch_size=4)

  wavconfig = Wav2Vec2Config.from_pretrained("superb/wav2vec2-base-superb-ks")
  vivitconfig = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
  vivitconfig.num_frames=10
  vivitconfig.video_size=[10,224,224]
  model = MultiModalClassification(8, wavconfig, vivitconfig)
  
  num_trainable_params = sum(p.numel() for p in model.audio_extractor.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in model.video_extractor.parameters() if p.requires_grad)
  print(f"Total number of trainable parameters: {num_trainable_params}")
  
  model.to(device)
  
  training_args = TrainingArguments(
            output_dir=f".../outputs/multilabel/multimodal",         
            num_train_epochs=20,             
            per_device_train_batch_size=30, 
            # gradient_accumulation_steps=2,
            per_device_eval_batch_size=30,    
            learning_rate=5e-4,#1e-5
            lr_scheduler_type="cosine_with_min_lr",
            logging_steps=10,                
            seed=42,                       
            evaluation_strategy="steps",  
            eval_steps=50,
            # save_strategy="epoch",  
            warmup_steps=10,      
            optim="sgd",          
            lr_scheduler_kwargs={"min_lr": 5e-6},    
            fp16=True,    
            metric_for_best_model="accuracy",
            load_best_model_at_end=True,
            report_to=None,
            save_total_limit=2,           
        )
  STUTTER_CLASSES = ['SR', 'ISR', 'MUR', 'P', 'B', 'V', 'FG', 'HM']
  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    y_pred = torch.from_numpy(predictions)
    y_true = torch.from_numpy(labels)
    y_pred = (y_pred.sigmoid() > 0.5)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_any = f1_score((torch.sum(y_true,dim=1)>0).float(), (torch.sum(y_pred,dim=1)>0).float(), average='macro')
    metrics = {}
    for i,classes in enumerate(STUTTER_CLASSES):
      metrics[classes] = f1_score(y_true[:,i], y_pred[:,i])
    return {'accuracy': (y_pred==y_true.bool()).float().mean().item(), 
            'f1_weighted': f1_weighted, 'f1_macro': f1_macro, 'f1_any':f1_any, **metrics}

  trainer = Trainer(
              model=model,                      
              args=training_args, 
              train_dataset=test_dataset,      
              eval_dataset=test_dataset,       
              compute_metrics=compute_metrics
          )
  
  # trainer.train()
  trainer.save_model("../outputs/multilabel/multimodal/")
  trainer.evaluate()
  
