from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import torch
import datasets as datasets_utils

def cont_training_loop(model, optimizer, criterion, datasets, device, batch_size=256, compensate_error=False):
    model = model.to(device)
    for j in range(len(datasets)):
        print(f'loading {datasets[j].name}')
        for i, (x, y) in enumerate((pbar := tqdm(DataLoader(datasets[j], batch_size=batch_size, shuffle=True), desc="Loss: N/A", leave=False))):
            x, y = x.to(device).float(), y.to(device).float()
            
            y_pred = model(x)
            
            if compensate_error:
                y = compensate_error(y_pred.detach().numpy(), y) # does not backprop through compensated vectors
                
            loss = criterion(y_pred, y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                pbar.set_description(f"Loss: {loss}")
    
def compensate_error(y_pred, y):
    dif = (y - y_pred).cumsum(axis=0)
    return np.vstack([y[0], y[1:] + dif[:-1]])
    
