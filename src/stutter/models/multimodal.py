from transformers import Wav2Vec2ForSequenceClassification, VivitForVideoClassification, VivitConfig
import torch
import torch.nn as nn

from typing import Optional


CACHE_DIR="/tmp/"
_HIDDEN_STATES_START_POSITION = 2


import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()
        
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualStack(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal)] * self._num_residual_layers)
        
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

class Conv1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv

class ConvolutionalEncoder(nn.Module):
    '''
        Taken from: https://github.com/swasun/VQ-VAE-Speech
    '''
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
        use_kaiming_normal, input_features_type, features_filters, sampling_rate,
        device, verbose=False):

        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=2
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )
        
        self._input_features_type = input_features_type
        self._features_filters = features_filters
        self._sampling_rate = sampling_rate
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):

        x_conv_1 = F.relu(self._conv_1(inputs))
        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        x_conv_3 = F.relu(self._conv_3(x))
        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        x = self._residual_stack(x_conv_5) + x_conv_5

        return x

class AudioExtractor(nn.Module):
    __acceptable_params__ = ['input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout']
    def __init__(self, **kwargs):
        super(AudioExtractor, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params__]
        
        # self.emb = nn.Linear(self.input_size, self.hidden_size)
        self.lstm1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, batch_first=True, dropout=self.dropout)
        self.lstm2 = nn.LSTM(self.hidden_size, 32, num_layers=1, batch_first=True, dropout=self.dropout)
        self.ln = nn.LayerNorm(32)
        self.emb = nn.Linear(32, self.output_size)
        
        
    def forward(self, x: torch.Tensor, tasks=['t1', 't2']):
        # x->(batch_size, seq_len, input_size)
        x = x.permute(0, 2, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm1(x, (h0, c0))
        out, _ = self.lstm2(out)

        out = self.ln(out[:,-1,:])
        
        return self.emb(out)

class VideoExtractor(VivitForVideoClassification):
    def __init__(self, config):
        super().__init__(config)
        self.vivit.requires_grad_(False)
        self.vivit.embeddings.requires_grad_(True)
        self.vivit.encoder.layer[0].requires_grad_(True)
        self.vivit.encoder.layer[11].requires_grad_(True)
        self.classifier.requires_grad_(True)
        self.emb = nn.Linear(768, 128)
        
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
        sequence_output = self.emb(sequence_output)
        return sequence_output

class MultiModalClassification(nn.Module):
    __acceptable_params__ = ['output_size', 'num_frames', 'video_size']
    def __init__(self, **kwargs):
        super().__init__()
        [setattr(self, k, kwargs.get(k, None)) for k, v in kwargs.items() if k in self.__acceptable_params__]

        self.audio_extractor  = AudioExtractor(input_size=40, hidden_size=64, num_layers=1, output_size=128, dropout=0.5)
        vivit_config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
        vivit_config.num_frames=self.num_frames
        vivit_config.video_size=self.video_size
        self.video_extractor = VideoExtractor.from_pretrained("google/vivit-b-16x2-kinetics400", config=vivit_config, ignore_mismatched_sizes=True, cache_dir=CACHE_DIR)
        self.activation = nn.GELU()
        self.projector = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, self.output_size)
        self.audio_layer_norm = nn.LayerNorm(128)
        self.video_layer_norm = nn.LayerNorm(128)
        self.multi_norm = nn.BatchNorm1d(128)
      

      
    def forward(self, pixel_values, input_values, labels):
        audio = self.audio_extractor(input_values)
        audio = self.audio_layer_norm(audio)
        video = self.video_extractor(pixel_values)
        video = self.video_layer_norm(video)
        x = torch.cat((video, audio), dim=1)
        output = self.projector(x)
        output = self.multi_norm(output)
        logits = self.classifier(self.activation(output))

        loss = None
        # if labels is not None:
        #   loss_fct = CrossEntropyLoss()
        #   loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if labels is not None and self.output_size == 1:
            logits = torch.sigmoid(logits).view(-1)
            loss_ce = torch.nn.BCELoss()
            loss = loss_ce(logits, labels)
        else:
            pos_weight = torch.tensor([1349/185, 1384/150, 1481/53, 1225/309, 1134/500,1354/180, 924/610 , 1346/188]).to(labels.device)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits.view(-1, self.output_size), 
                            labels.float().view(-1, self.output_size))

        return (loss, logits)
        

if __name__ == "__main__":

    kwargs = {
        'output_size': 8,
        'vid_num_frames': 10,
        'vid_size': [10, 224, 224]
    }
    model = MultiModalClassification(**kwargs)
    print(model)
    audio_input = torch.randn(32, 1500, 40)
    video_input = torch.randn(32, 10, 3, 224, 224)
    labels = torch.randint(0, 2, (32, 8))
    loss, logits = model(video_input, audio_input, labels)

