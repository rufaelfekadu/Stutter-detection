
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Config
import torchaudio
import torch
import torch.nn as nn

_HIDDEN_STATES_START_POSITION = 2


class AudioEncoder(Wav2Vec2ForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.freeze_base_model()
        # self.wav2vec2.feature_extractor.requires_grad_(True)
        # self.projector.requires_grad_(True)
        self.fc = nn.Linear(256,512)
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
        return self.fc(pooled_output)

class Wav2Vec2Classifier(nn.Module):
    __acceptable_parameters = ['hidden_size', 'output_size']
    def __init__(self, **kwargs):
        super().__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_parameters]
        encoder_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
        self.encoder = AudioEncoder(encoder_config)

        self.fc = nn.Sequential(
                    nn.Linear(self.hidden_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU()
                    )
        self.fc1 = nn.Linear(64, 1)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(64, self.output_size)

    def forward(self, x, tasks=['t1']):
        x = self.encoder(x)
        out_t1, out_t2 = None, None
        x = self.fc(x)
        if 't1' in tasks:
            out_t1 = self.fc1(x)
        if 't2' in tasks:
            out_t2 = self.fc2(x)
        return self.act(out_t1), out_t2


if __name__ == '__main__':
    model = Wav2Vec2Classifier(feature_size=512, output_size=5)
    x = torch.randn(1, 90000)
    y = model(x)
    print(y.shape)