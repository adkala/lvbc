from torch import nn

import torch


class HorizonLSTM(nn.Module):  # window and horizon treated same
    def __init__(self, num_layers=4, input_size=13, hidden_size=200, output_size=3):
        super().__init__()
        self.inputl = nn.Sequential(
            nn.Linear(input_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), hidden_size),
        )
        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.outputl = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), output_size),
        )

    def forward(self, x):
        x = self.inputl(x)
        x, _ = self.lstm(x)
        x = self.outputl(x)
        return x

    def generate(self, x, u):
        x = self.inputl(x)
        x, (c, h) = self.lstm(x)
        x = self.outputl(x)

        y = [x[-1]]
        for i in range(u.shape[0]):
            x = torch.hstack([y[-1], u[i]]).unsqueeze(0)
            x = self.inputl(x)
            x, (c, h) = self.lstm(x, (c, h))
            x = self.outputl(x)
            y.append(x[-1])
        return torch.vstack(y)


class HorizonLSTMWithVariance(HorizonLSTM):
    def __init__(self, num_layers=4, input_size=13, hidden_size=200, output_size=3):
        super().__init__(num_layers, input_size, hidden_size, output_size)
        self.outputl_var = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), output_size),
            nn.ELU(),
        )

    def forward(self, x):
        x = self.inputl(x)
        x, _ = self.lstm(x)

        mean = self.outputl(x)
        var = self.outputl_var(x)

        return mean, var

    def generate(self, x, u):
        raise NotImplementedError
