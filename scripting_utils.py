from utils import MATELoss
from models import HorizonLSTM, WindowHorizonLSTM
from datasets import LSTMDataset, set_global_normalization_factors

import torch
import os
import pickle

def make_config(
        model,
        dataset,
        epochs: int = 400,
        train_path: str = 'data',
        test_path: str = 'data',
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
        num_layers: int = 4, 
        input_size: int = 11,
        hidden_size: int = 6,
        batch_size: int = 256,
        optimizer: str = 'adamw',
        learning_rate: float = 1e-3,
        loss_function: str = 'mate',
        on_track: bool = True,
        use_mask: bool = True,
        ):

    def get_make_model_config(name):
        if name == 'horizonlstm':
            return make_horizonlstm_config
        raise ValueError

    def get_make_dataset_config(name):
        if name == 'lstmdataset':
            return make_lstmdataset_config
        raise ValueError

    def make_horizonlstm_config():
        # model
        model = HorizonLSTM(num_layers=num_layers, input_size=input_size, output_size=hidden_size)

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
            'batch_size': batch_size,
            'on_track': on_track,
            'include_pos': include_pos, # for on track
            'delta_p': delta_p, # for on track
            'use_mask': use_mask,
            'epochs': epochs
        }
        
    def make_lstmdataset_config():
        def get_datasets(path):
            files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.p')]

            unpickled = []
            for fp in files:
                with open(fp, 'rb') as f:
                    unpickled.append(pickle.load(f))
            
            datasets = []
            for ds in unpickled:
                datasets.append(LSTMDataset(ds, window=window, horizon=horizon, delta_p=delta_p, include_pos=include_pos, include_r=include_r, no_z=no_z, zero_position=zero_position, mask_window=mask_window))

            if normalize:
                set_global_normalization_factors(datasets)
            
            return datasets

        train_datasets = get_datasets(train_path) 
        test_datasets = get_datasets(test_path)

        return {
            'train_datasets': train_datasets,
            'test_datasets': test_datasets,
            'window': window,
            'horizon': horizon,
            'bagging': bagging,
            'no_z': no_z,
        }
    
    return get_make_model_config(model)() | get_make_dataset_config(dataset)()
    