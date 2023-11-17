from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import numpy as np
import torch
import datasets as datasets_utils

class MATELoss: # mean absolute trajectory error (RPE if delta_p)
    def __init__(self, eps=1e-6, ret_ind_time_steps=False): 
        self.mse = nn.MSELoss(reduction="none")
        self.eps = eps
        self.ret_ind_time_steps = ret_ind_time_steps

    def __call__(self, y_pred, y, mask=None) -> (torch.Tensor, torch.Tensor):
        if mask == None:
            mask = np.ones(y.shape[:2])

        transformed_mask = mask.bool().unsqueeze(-1).expand(-1, -1, y.shape[-1])
        y_pred = y_pred[:,:,-y.shape[-1]:] * transformed_mask
        y *= transformed_mask
        dist = (self.mse(y_pred, y).sum(dim=2)  + self.eps).sqrt()

        total_loss = dist.sum() / mask.sum()
        ind_loss = dist.sum(axis=1) / mask.sum(axis=1)

        ind_loss[ind_loss == np.inf] = 0

        return total_loss, ind_loss

    
def training_loop(config):
    model, optimizer, criterion, datasets, device = config['model'], config['optimizer'], config['criterion'], config['train_datasets'], config['device']

    total_loss = torch.zeros(config['window'] + config['horizon']) # y len
    total = torch.zeros(config['window'] + config['horizon']) # y len

    for j in range(len(datasets)):
        i = j
        if config['bagging']:
            i = np.random.randint(len(datasets))

        print(f'running {datasets[i].name}')

        pbar = tqdm(DataLoader(datasets[i], batch_size=config['batch_size'], shuffle=True, collate_fn=datasets_utils.make_lstm_collate_fn(config)), desc="Loss: N/A", leave=False)
        for i, (x, y, m) in enumerate(pbar):
            x, y, m = x.to(device), y.to(device), m.to(device)

            pos_len = 2 if config['no_z'] else 3

            if config['on_track'] and config['include_pos']:
                out, ch = model(x[:config["window"] + 1])
                k = 1
                out_l = [out]
                while k < config['horizon']:
                    new_x = torch.unsqueeze(torch.hstack([x[k + config['window'], :, :-pos_len], out_l[-1][-1, :, -pos_len:] + (x[k + config['window'] - 1, :, -pos_len:] if config['delta_p'] else 0)]), dim=0) # add (or replace) last val with predicted val in trajectory
                    out, ch = model(new_x, ch)
                    out_l.append(out)
                    k += 1
                y_pred = torch.vstack(out_l)
            else:
                y_pred, _ = model(x)
            y_pred = y_pred[:,:,-pos_len:]

            if config['on_track'] and config['delta_p']:
                tm = m.bool().unsqueeze(-1).expand(-1, -1, y.shape[-1])
                y = get_corrected_y_for_delta_p(y, y_pred, tm)

            loss, loss_ind = criterion(y_pred, y, m) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            item_count = m.sum(axis=1)
            total_loss += loss_ind.detach() * item_count
            total += item_count
            
            if i % 10 == 0:
                pbar.set_description(f"Loss: {loss}")

    loss = total_loss.sum() / total.sum()
    loss_ind = total_loss / total

    loss_ind[loss_ind != loss_ind] = 0 # catching nan due to 0 / 0

    return loss, loss_ind

def get_corrected_y_for_delta_p(y, y_pred, m):
    dif = (y - y_pred).cumsum(axis=0)
    y[1:] += dif[:-1]
    y *= m
    return y