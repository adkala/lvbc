from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

import numpy as np
import pickle
import torch

class PolyLSTM(nn.Module): # big issue with magnitudes of mantissa and exponent of coefficients being vastly different
    def __init__(self):
        super().__init__()

        self.mlp1 = nn.Linear(6, 200)
        self.lstm1 = nn.LSTM(4, 200, 200, batch_first = True)
        self.mlp2 = nn.Sequential(
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, 24)
        )

    def forward(self, x):
        x = self.mlp1(x)
        x, _ = self.lstm1(x)
        x = self.mlp2(x)

        return x

class PolyMLP(nn.Module):
    def __init__(self, input_size=600, output_size=24, hidden_size=300, hidden_layers=2):
        super().__init__()
        
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Sequential(*([nn.Linear(hidden_size, hidden_size), nn.ReLU()] * hidden_layers))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.input(x)
        x = self.hidden(x)
        x = self.out(x)
        return x

class PolyDataset(Dataset):
    def __init__(self, bag, degree=11, window=100, horizon=100, normalize_p=True, frexp=True, mlp=True, gt=False):
        self.p = np.vstack(bag['p'])
        self.v = np.vstack(bag['v'])

        self.window, self.horizon = window, horizon
        self.degree = degree
        self.gt = gt
        self.mlp = mlp
        self.frexp = frexp
        
        if normalize_p:
            self.set_normalize_p_factors(*self.get_normalize_p_factors())
    
    def __len__(self):
        return self.p.shape[0] - self.window - self.horizon
    
    def __getitem__(self, i):
        x = np.hstack([self.p[i:i+self.window], self.v[i:i+self.window]])
        if self.mlp:
            x = x.flatten()

        y_gt = self.p[i+self.window:i+self.window+self.horizon] 
        t = np.linspace(0, 99, 100)
        x_poly = np.polyfit(t, y_gt[:, 0], self.degree)
        y_poly = np.polyfit(t, y_gt[:, 1], self.degree)
        if self.frexp:
            x_poly = np.hstack(np.frexp(x_poly))
            y_poly = np.hstack(np.frexp(y_poly))
        y = np.hstack([x_poly, y_poly])

        if self.gt:
            return x, y, y_gt
        else:
            return x, y
    
    def set_normalize_p_factors(self, m, r):
        self.p = (self.p - m) / r

    def get_normalize_p_factors(self):
        mi, ma = np.min(self.p, axis=0), np.max(self.p, axis=0)
        return mi, ma - mi

def training_loop(model, optimizer, criterion, dataset, batch_size, device, update_loss_iter=10):
    dl = DataLoader(dataset, batch_size=batch_size)
    pbar = tqdm(dl, desc='Loss: N/A', leave=False)

    total_loss = 0
    total = 0
    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device).float(), y.to(device).float()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += 1

        if i % update_loss_iter == 0:
            pbar.set_description(f'Loss: {loss.item()}')
    return total_loss / total

def main():
    epochs = 10000
    batch_size=256
    learning_rate=1.e-3
    data_file = 'data/test0_filtered/rb15_08_15-16_28_set_0.p'
    save_file = 'models/polyfitmlp.p'
    
    model = PolyMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    model.to(device)

    with open(data_file, 'rb') as f:
        bag = pickle.load(f)
    dataset = PolyDataset(bag, mlp=True)

    model.train()
    loss_history = []
    for e in (pbar := tqdm(range(epochs), desc="Epoch Loss: N/A")):
        loss = training_loop(model, optimizer, criterion, dataset, batch_size, device)

        loss_history.append(loss)
        if e % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'loss_history': loss_history,
                'epochs': e
                       }, save_file)

        pbar.set_description(f'Epoch Loss: {loss}')

if __name__ == "__main__":
    main()

    



