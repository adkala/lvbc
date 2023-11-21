from torch import nn

import torch

class HorizonLSTM(nn.Module): # window and horizon treated same
    def __init__(self, num_layers=4, input_size=13, hidden_size=200, output_size=3):
        super().__init__()
        self.inputl = nn.Sequential(
            nn.Linear(input_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), hidden_size)
        )
        self.lstm = nn.LSTM(num_layers=num_layers, input_size=hidden_size, hidden_size=hidden_size)
        self.outputl = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), output_size)
        )

    def forward(self, x):
        x = self.inputl(x)
        x, _ = self.lstm(x)
        x = self.outputl(x)
        return x