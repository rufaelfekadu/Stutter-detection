import torch

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, tasks=['t1', 't2']):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x, (h0, c0))
        t1_out, t2_out = None, None

        if 't1' in tasks:
            t1_out = self.fc1(out[:,-1,:])
        if 't2' in tasks:
            t2_out = self.fc2(out[:,-1,:])

        return t1_out, t2_out
    
       