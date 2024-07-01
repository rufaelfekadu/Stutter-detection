import torch

import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 2)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor, task_type='mtl'):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x, (h0, c0))
        if task_type == 'mtl':
            return  nn.functional.log_softmax(self.fc1(out[:,-1,:]), dim=1), self.fc2(out[:,-1,:])
        elif task_type == 't1':
            return nn.functional.log_softmax(self.fc1(out[:,-1,:]), dim=1), None
        else:
            return None, self.fc2(out[:,-1,:])