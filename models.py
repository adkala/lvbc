from torch import nn

import torch

class HorizonLSTM(nn.Module): # window and horizon treated same
    def __init__(self, num_layers=4, input_size=9, hidden_size=6): # set input size to 12 if not using r
        super().__init__()
        self.lstm1 = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, ch=None):
        x, _ = self.lstm1(x, ch)
        return x

class HorizonWithHeadLSTM(nn.Module): # window and horizon treated same
    def __init__(self, num_layers=4, input_size=9, hidden_size=6): # set input size to 12 if not using r
        super().__init__()
        #self.lstm1 = nn.LSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size)
        #self.head = nn.Sequential(
        #    nn.Linear(6, 32),
        #    nn.ReLU(),
        #    nn.Linear(32, 2),
        #    #nn.Tanh(),
        #    #nn.Linear(6, 2)
        #)

        self.mlp1 = nn.Linear(9, 9)
        self.lstm1 = nn.LSTM(num_layers=5, input_size=9, hidden_size=256)
        self.dropout1 = nn.Dropout(p=0.5)
        self.lstm2 = nn.LSTM(num_layers=5, input_size=256, hidden_size=256)
        self.dropout2 = nn.Dropout(p=0.5)
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x, ch=None):
        #x, ch = self.lstm1(x, ch)
        #return self.head(x)
        
        x = self.mlp1(x)
        x, (c, h) = self.lstm1(x)
        x, c, h = self.dropout1(x), self.dropout1(c), self.dropout1(h)
        x, _ = self.lstm2(x, (c, h))
        return self.mlp2(x)

    def generate(self, x):
        pass

class MLPEncLSTMDec(nn.Module):
    def __init__(self, x_input_size=400, u_input_size=7, dec_hidden_size=256, lstm_num_layers=8):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(x_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, dec_hidden_size)
        )
        self.dropout1 = nn.Dropout(p=0.5)

        self.mlp2 = nn.Sequential(
            nn.Linear(u_input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, dec_hidden_size)
        )
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.hidden_state = nn.Parameter(torch.zeros((lstm_num_layers, dec_hidden_size)))
        self.lstm1 = nn.LSTM(num_layers=lstm_num_layers, input_size=dec_hidden_size, hidden_size=dec_hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.mlp3 = nn.Sequential(
            nn.Linear(dec_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 3)
        )

        self.lstm_num_layers = 8

    def forward(self, x, u):
        x = self.mlp1(x)
        x = self.dropout1(x)

        u = self.mlp2(u)
        u = self.dropout2(u)

        h, c = torch.vstack([torch.unsqueeze(self.hidden_state, dim=0).contiguous()] * x.shape[0]).transpose(0, 1), torch.vstack([torch.unsqueeze(x, dim=0)] * self.lstm_num_layers)
        x, (h, c) = self.lstm1(u, (h, c))
        x, h, c = self.dropout3(x), self.dropout3(h), self.dropout3(c)
        
        return self.mlp3(x)
