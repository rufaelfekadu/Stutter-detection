import torch

import torch.nn as nn

class LSTMModel(nn.Module):
    __acceptable_params__ = ['input_size', 'hidden_size', 'num_layers', 'output_size', 'dropout']
    def __init__(self, **kwargs):
        super(LSTMModel, self).__init__()
        [setattr(self, k, v) for k, v in kwargs.items() if k in self.__acceptable_params__]
        
        # self.emb = nn.Linear(self.input_size, self.hidden_size)
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        self.fc1 = nn.Linear(self.hidden_size, 2)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x: torch.Tensor, tasks=['t1', 't2']):
        # x->(batch_size, seq_len, input_size)
        # x = x.permute(0, 2, 1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # x = self.emb(x)

        out, _ = self.lstm(x, (h0, c0))
        t1_out, t2_out = None, None
        
        if 't1' in tasks:
            t1_out = self.fc1(out[:, -1, :])
        if 't2' in tasks:
            t2_out = self.fc2(out[:, -1, :])

        return t1_out, t2_out
    

if __name__ == "__main__":
    kwargs = {
        'input_size': 40,
        'hidden_size': 64,
        'num_layers': 1,
        'output_size': 6,
        'dropout': 0.5
        }
    model = LSTMModel(**kwargs)
    x = torch.rand(32, 301, 40)
    t1_out, t2_out = model(x)
    print(t1_out.shape, t2_out.shape)
    print(model)