from torch.utils.data import Dataset
from torch import nn

import torch
import numpy as np

# all defaults are made with assumed 20hz sampling

class BaseDataset(Dataset):
    def __init__(self, bag, start=0, end=None):
        if not end:
            end = len(bag['t'])
        
        self.name = bag['name']
        self.r = bag['r'][start:end]
        self.v = bag['v'][start:end] # values come from position difference (with processing)
        self.p = bag['p'][start:end]
        self.u = bag['u'][start:end]

    def __len__(self):
        return len(self.r)
    
    def __getitem__(self, i):
        return self.r[i], self.v[i], self.p[i], self.u[i]

class LSTMDataset(BaseDataset):
    def __init__(self, bag, start=0, end=None, window=100, horizon=100, rotate_acceleration=True, normalize_self=False, delta_p = True, include_pos=True, include_r=True, no_z=True, zero_position=False, mask_window=True): # 5 sec window, 5 sec horizon
        super().__init__(bag, start=start, end=end)

        self.r = np.array(self.r)
        self.p = np.array(self.p)
        self.u = np.array(self.u)

        if rotate_acceleration: # rotating acceleration here since always known
            self.u[:, :3] = np.vstack([self.r[i] @ self.u[i, :3] for i in range(self.u.shape[0])])

        self.window = window # used as 'precursor'
        self.horizon = horizon

        self.u_mean, self.u_std = None, None
        self.u_mean_no_z, self.u_std_no_z = None, None

        self.set_getitem_params(delta_p, include_pos, include_r, no_z, zero_position, mask_window)

        if normalize_self:
            self.normalize_self()

    def __len__(self):
        return self.r.shape[0] - self.window - 1

    def __getitem__(self, i):
        '''
        Options explanation \n
        - delta_p: set output to delta_p (default True)
        - include_pos: include_pos in x vector (default True)
        - include_r: include r in x vector (default True)
        - no_z: include z in vectors (default True)
        - mask_window: mask window during training (default True)
        - zero_position: set first position in seq to be the seq's origin (default False, only valid if include_pos])
        '''
        # set params
        delta_p, include_pos, zero_position, include_r, no_z, mask_window = self.delta_p, self.include_pos, self.zero_position, self.include_r, self.no_z, self.mask_window

        j = i + self.window + self.horizon 

        r, p, u = np.copy(self.r[i:j]), np.copy(self.p[i:j]), np.copy(self.u[i:j])

        if no_z:
            u = np.hstack([u[:, :2], u[:, 3:]])
            r = r[:, :-1, :-1]
            p = p[:, :-1]
            

        if self.u_mean is None:
            print('Normalization factors not set. Continuing with unnormalized values.')
        else: 
            if no_z:
                u = (u - self.u_mean_no_z) / self.u_std_no_z
            else:
                u = (u - self.u_mean) / self.u_std

        dp = np.copy(p)
        dp[1:] -= dp[:-1] # delta p
        dp[0, :] = 0

        if zero_position:
            p -= p[0]

        x = np.hstack([u, 
                       r.reshape(r.shape[0], -1) if include_r else np.empty((u.shape[0], 0)), 
                       p if include_pos else np.empty((u.shape[0], 0))
                       ])
        y = dp if delta_p else p

        mask = np.sum(dp, axis=-1) != 0 # index to include 
        mask[:self.window] = 0 if mask_window else 1

        return torch.from_numpy(x).float(), torch.from_numpy(y).float(), torch.from_numpy(mask).bool()

    def set_getitem_params(self, delta_p = True, include_pos=True, include_r=True, no_z=True, zero_position=False, mask_window=True):
        self.delta_p, self.include_pos, self.include_r, self.no_z, self.zero_position, self.mask_window= delta_p, include_pos, include_r, no_z, zero_position, mask_window

    def set_normalization_factors(self, u_mean, u_std):
        self.u_mean, self.u_std = u_mean, u_std
        self.u_mean_no_z = np.hstack([u_mean[:2], u_mean[3:]])
        self.u_std_no_z = np.hstack([u_std[:2], u_std[3:]])

    def get_normalization_factors(self):
        return np.mean(self.u, axis=0), np.std(self.u, axis=0), self.u.shape[0]
    
    def normalize_self(self):
        self.set_normalization_factors(*self.get_normalization_factors()[:-1])


def set_global_normalization_factors(datasets):
    u_m, u_v, t = 0, 0, 0
    for d in datasets:
        _u_m, _u_s, _t = d.get_normalization_factors()
        u_m += _u_m * _t
        u_v += _u_s ** 2 * _t
        t += _t
    u_m /= t
    u_s = np.sqrt(u_v / t)
    for d in datasets:
        d.set_normalization_factors(u_m, u_s)

def make_lstm_collate_fn(config):
    def lstm_collate_fn(batch): # batch in form L, x, want l, h, x
        x, y, m = zip(*batch)

        size = config['window'] + config['horizon']

        x = torch.cat([torch.unsqueeze(
            torch.vstack([_x, torch.zeros((size - _x.shape[0], _x.shape[1]))])
        , dim=0) for _x in x]).transpose(0, 1)
        y = nn.utils.rnn.pad_sequence(y)
        m = torch.hstack([torch.unsqueeze(torch.hstack([_m, torch.ones(y.shape[0] - _m.shape[0])]), dim=-1) for _m in m])

        return x, y, m
    return lstm_collate_fn