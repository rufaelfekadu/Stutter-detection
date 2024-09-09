import torch
import torch.nn as nn
from transformers import VivitConfig, VivitForVideoClassification


class VivitForStutterClassification(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        config.num_frames=cfg.model.vivit.num_frames
        config.video_size=cfg.model.vivit.video_size
        vivit = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400", 
                                                            config=config, 
                                                            ignore_mismatched_sizes=True, 
                                                            cache_dir="/tmp/")
        self.vivit = vivit
        self.vivit.requires_grad_(False)
        self.vivit.vivit.encoder.layer[0].requires_grad_(True)
        self.vivit.vivit.encoder.layer[11].requires_grad_(True)
        self.vivit.classifier.requires_grad_(True)
        self.activation = nn.GELU()
        self.fc1 = torch.nn.Linear(400, 400)
        self.num_labels = cfg.model.output_size
        self.y = torch.nn.Linear(400, cfg.model.output_size)
        self.pos_weight = torch.tensor([1354/180, 924/610 , 1346/188])
        self.cfg = cfg
        
    def forward(self, pixel_values, labels):
        outputs = self.vivit.vivit(pixel_values)
        logits = self.vivit.classifier(outputs[0][:, 0, :])
        logits = self.fc1(logits)
        logits = self.y(self.activation(logits))
        loss = None
        if labels is not None and self.cfg.tasks[0] == "t1":
            logits = torch.sigmoid(logits).view(-1)
            loss_ce = torch.nn.BCELoss()
            loss = loss_ce(logits, labels)
            
        elif labels is not None:
            loss_bce = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight.to(labels.device))
            loss = loss_bce(logits.view(-1, self.num_labels), 
                            labels.float().view(-1, self.num_labels))
            
        return (loss, logits)