from torch import nn

import utils
import models
import datasets as datasets_utils

import torch
import os
import pickle

def get_make_model_config(name):
    if name == 'horizonlstm':
        return make_horizonlstm_config
    raise ValueError


def get_make_dataset_config(name):
    if name == 'contdataset':
        return make_contdataset_config
    raise ValueError


def make_horizonlstm_config(
        num_layers: int = 4, 
        input_size: int = 13,
        hidden_size: int = 200,
        output_size: int = 3,
        \
        optimizer: str = 'adamw',
        learning_rate: float = 1.e-5,
        loss_function: str = 'mse',
        device: str = 'default',
        \
        epochs: int = 10000,
        batch_size: int = 256,
        compensate_error: bool = False,
        delta_p: bool = True,
        **kwargs
        ):

    # model
    model = models.HorizonLSTM(num_layers=num_layers, input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # device
    if device == 'default':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f'using {device}')

    model.to(device)

    # optimizer
    if optimizer == 'adamw':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError

    # loss_function
    if loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError
    
    # compensate_error check
    if compensate_error and not delta_p:
        raise ValueError('compensate_error cannot be True when delta_p is False')
    elif compensate_error:
        raise NotImplementedError
    
    training_loop = lambda datasets: utils.cont_training_loop(model, optimizer, criterion, datasets, device, batch_size=batch_size, comp_error=compensate_error)
    
    return {
        'epochs': epochs,
        'model': model,
        'optimizer': optimizer,
        'training_loop': training_loop,
        'device': device
    }
        
def make_contdataset_config(
    train_path: str = 'data',
    validation_path: str = 'data',
    window: int = 1,
    horizon: int = 100,
    delta_p: bool = True,
    normalize_p: bool = True,
    standardize_u: bool = True,
    **kwargs
):
    def get_datasets(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.p')]
        datasets = []
        for fp in files:
            with open(fp, 'rb') as f:
                datasets.append(datasets_utils.ContDataset(pickle.load(f), window=window, horizon=horizon, delta_p=delta_p))
        return datasets

    train_datasets = get_datasets(train_path)
    validation_datasets = get_datasets(validation_path)

    if normalize_p:
        datasets_utils.set_global_p_normalization_factors(train_datasets + validation_datasets)
    if standardize_u:
        datasets_utils.set_global_u_and_v_standardization_factors(train_datasets + validation_datasets)

    return {
        'train_datasets': train_datasets,
        'validation_datasets': validation_datasets,
        'window': window,
        'horizon': horizon,
        'delta_p': delta_p,
        'normalize_p': normalize_p,
        'standardize_u': standardize_u
    }
    
