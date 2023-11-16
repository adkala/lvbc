from torch import nn

import torch

class HorizonLSTM(nn.Module): # window and horizon treated same
    def __init__(self, num_layers=2, input_size=11, output_size=2): # set input size to 12 if not using r
        super().__init__()
        self.lstm1 = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=output_size)

    def forward(self, x, ch=None):
        return self.lstm1(x, ch)

    def generate(self, x):
        pass
        
    
class WindowHorizonLSTM(nn.Module): # window and horizon treated differently 
    def __init__(self, num_layers_1=1, num_layers_2=1, input_size=11, hidden_size=6, output_size=2):
        pass