from utils import MATELoss
from models import HorizonLSTM, HorizonWithHeadLSTM, MLPEncLSTMDec
from datasets import LSTMDataset, EncDecDataset, set_global_normalization_factors, set_global_standardization_factors

import torch
import os
import pickle

def get_make_model_config(name):
    if name == 'horizonlstm':
        return make_horizonlstm_config
    elif name == 'horizonwithheadlstm':
        return make_horizonwithheadlstm_config
    elif name == 'mlpenclstmdec':
        return make_mlpenclstmdec_config
    raise ValueError


def get_make_dataset_config(name):
    if name == 'lstmdataset':
        return make_lstmdataset_config
    if name == 'encdecdataset':
        return make_encdecdataset_config
    raise ValueError


def make_horizonlstm_config(
        epochs: int = 400,
        num_layers: int = 4, 
        input_size: int = 9,
        hidden_size: int = 6,
        batch_size: int = 256,
        optimizer: str = 'adamw',
        learning_rate: float = 1e-3,
        loss_function: str = 'mate',
        on_track: bool = True,
        **kwargs
        ):

    # model
    model = HorizonLSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size)

    # optimizer
    if optimizer == 'adamw':
        optimizer_obj = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss_function
    if loss_function == 'mate':
        criterion = MATELoss()

    return {
        'model': model,
        'optimizer': optimizer_obj,
        'criterion': criterion,
        'epochs': epochs,
        'on_track': on_track,
    }

def make_horizonwithheadlstm_config(
        epochs: int = 400,
        num_layers: int = 4, 
        input_size: int = 9,
        hidden_size: int = 6,
        batch_size: int = 256,
        optimizer: str = 'adamw',
        learning_rate: float = 1e-3,
        loss_function: str = 'mate',
        on_track: bool = True,
        **kwargs
        ):

    # model
    model = HorizonWithHeadLSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size)

    # optimizer
    if optimizer == 'adamw':
        optimizer_obj = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss_function
    if loss_function == 'mate':
        criterion = MATELoss()

    return {
        'model': model,
        'optimizer': optimizer_obj,
        'criterion': criterion,
        'epochs': epochs,
        'on_track': on_track,
    }

def make_mlpenclstmdec_config(
    epochs: int=400,
    batch_size: int=256,
    optimizer: str = 'adamw',
    learning_rate: float = 1e-3,
    loss_function: str = 'mate',
    **kwargs
):
    # model
    model = MLPEncLSTMDec()

    # optimizer
    if optimizer == 'adamw':
        optimizer_obj = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # loss_function
    if loss_function == 'mate':
        criterion = MATELoss()

    return {
        'model': model,
        'optimizer': optimizer_obj,
        'criterion': criterion,
        'epochs': epochs,
    }
    
        
def make_lstmdataset_config(
    train_path: str = 'data',
    test_path: str = 'data',
    batch_size: int = 256,
    window: int = 100,
    horizon: int = 100,
    normalize: bool = True,
    delta_p: bool = True,
    include_pos: bool = True,
    include_r: bool = True,
    no_z: bool = True,
    zero_position: bool = False,
    mask_window: bool = True,
    bagging: bool = False,
    **kwargs
):
    def get_datasets(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.p')]

        unpickled = []
        for fp in files:
            with open(fp, 'rb') as f:
                unpickled.append(pickle.load(f))
        
        datasets = []
        for ds in unpickled:
            datasets.append(LSTMDataset(ds, window=window, horizon=horizon, delta_p=delta_p, include_pos=include_pos, include_r=include_r, no_z=no_z, zero_position=zero_position, mask_window=mask_window))
        
        return datasets

    train_datasets = get_datasets(train_path) 
    test_datasets = get_datasets(test_path)
    if normalize:
        set_global_normalization_factors(train_datasets + test_datasets)

    return {
        'train_datasets': train_datasets,
        'test_datasets': test_datasets,
        'batch_size': batch_size,
        'window': window,
        'horizon': horizon,
        'delta_p': delta_p,
        'include_pos': include_pos,
        'bagging': bagging,
        'no_z': no_z,
    }

def make_encdecdataset_config(
    train_path: str = 'data',
    test_path: str = 'data',
    batch_size: int = 256,
    window: int = 100,
    horizon: int = 100,
    delta_p: bool = True,
    bagging: bool = False,
    **kwargs
):
    def get_datasets(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.p')]

        unpickled = []
        for fp in files:
            with open(fp, 'rb') as f:
                unpickled.append(pickle.load(f))
        
        datasets = []
        for ds in unpickled:
            datasets.append(EncDecDataset(ds))
        
        return datasets

    train_datasets = get_datasets(train_path) 
    test_datasets = get_datasets(test_path)

    set_global_standardization_factors(train_datasets + test_datasets)
    set_global_normalization_factors(train_datasets + test_datasets)

    return {
        'train_datasets': train_datasets,
        'test_datasets': test_datasets,
        'batch_size': batch_size,
        'window': window,
        'horizon': horizon,
        'delta_p': delta_p,
        'bagging': bagging,
    }
    