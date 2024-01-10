from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import torch
import datasets as datasets_utils

def cont_training_loop(model, optimizer, criterion, datasets, device, batch_size=256, comp_error=False):
    total_loss = 0
    total = 0
    for j in range(len(datasets)):
        print(f'loading {datasets[j].name}')
        for i, (x, y) in enumerate((pbar := tqdm(DataLoader(datasets[j], batch_size=batch_size, shuffle=True), desc="Loss: N/A", leave=False))):
            x, y = x.to(device).float(), y.to(device).float()
            
            y_pred = model(x)
            
            if comp_error:
                y = compensate_error(y_pred.detach(), y) # does not backprop through compensated vectors
                
            loss = criterion(y_pred, y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total += 1
            
            if i % 10 == 0:
                pbar.set_description(f"Loss: {loss.item()}")

    return total_loss / total

def validate_data(config, samples=10):
    config['model'].eval()
    ma, me = [], []
    for _ in range(samples):
        i = np.random.randint(len(config['validation_datasets']))
        j = np.random.randint(len(config['validation_datasets'][i]))
        
        x, y = config['validation_datasets'][i][j]
        y *= config['validation_datasets'][i].p_r

        y_pred = config['model'].generate(torch.tensor(x[:config['window']]).float(), torch.tensor(x[config['window']:, 3:]).float())
        y_pred = y_pred.detach().numpy() * config['validation_datasets'][i].p_r
    
        x = x[:, :3] * config['validation_datasets'][i].p_r
        y = y[-config['horizon']:]

        cfe = get_car_frame_error(y_pred, y, x[-config['horizon']:, -1])
        
        ma.append(np.max(np.abs(cfe), axis=0))
        me.append(np.mean(np.abs(cfe), axis=0))

    print('car frame error (max, mean):', sum(ma) / samples, sum(me) / samples, '\n')
    config['model'].train()
    
def compensate_error(y_pred, y):
    if isinstance(y_pred, torch.Tensor) or isinstance(y, torch.Tensor):
        dif = (y - y_pred).cumsum(dim=-2)
        return torch.vstack([torch.unsqueeze(y[0], dim=0), y[1:] + dif[:-1]])
    else:
        dif = (y - y_pred).cumsum(axis=-2)
        return np.vstack([y[0], y[1:] + dif[:-1]])
    
def get_car_frame_error(y_pred, y, r):
    rr = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]]).transpose(2, 0, 1)
    sr = (y - y_pred)[:, :2]
    return np.einsum('BNi, Bi->BN', rr, sr)