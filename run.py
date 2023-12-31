import scripting_utils
import utils

import argparse
import yaml
import os
import torch
import time

def make_config(config_file: str) -> dict:
    config_kwargs = {'save_iter': 200}
    with open(config_file, "r") as f:
        config_kwargs |= yaml.load(f, Loader=yaml.SafeLoader)

    config = scripting_utils.get_make_dataset_config(config_kwargs['dataset'])(**config_kwargs)
    sample_point = config['train_datasets'][0][0] # for shape matching
    config.update(scripting_utils.get_make_model_config(config_kwargs['model'])(
        input_size=sample_point[0].shape[-1], 
        output_size=sample_point[1].shape[-1] * (2 if config_kwargs['loss_function'] == 'gaussian' else 1), # heteroscedastic
        **config_kwargs))
    
    config['name'] = os.path.splitext(
        os.path.basename(config_file))[0] + \
        '_' + f'lr{config_kwargs["learning_rate"]}-l{config_kwargs["num_layers"]}-h{config_kwargs["hidden_size"]}-{config_kwargs["loss_function"]}' + \
        '_' + time.strftime('%d-%m-%Y_%H-%M-%S')

    config['save_iter'] = config_kwargs['save_iter']
    
    return config

def run_training_loop(config, load_model=None):
    # create folder
    if not load_model:
        if not os.path.exists(f'models/{config["name"]}'):
            os.makedirs(f'models/{config["name"]}')

        loss_history = []
        epochs = 0
    else:
        torch_dict = torch.load(load_model, map_location=config['device'])

        config['model'].load_state_dict(torch_dict['model_state_dict'])
        config['optimizer'].load_state_dict(torch_dict['optimizer_state_dict'])

        loss_history = torch_dict['train_loss_history']
        epochs = torch_dict['epoch']

        config['name'] = torch_dict['name']

    for e in range(epochs + 1, config['epochs']):
        print(f'Starting epoch {e} / {config["epochs"]}')
        loss = config['training_loop'](config['train_datasets'])
        loss_history.append(loss)
        if e % config['save_iter'] == 0:
            torch.save({
                'name': config['name'],
                'epoch': e,
                'model_state_dict': config['model'].state_dict(),
                'optimizer_state_dict': config['optimizer'].state_dict(),
                'train_loss_history': loss_history,
            }, f'models/{config["name"]}/{config["name"]}_e{e}.p')
        print(f'Epoch {e} loss: {loss} \n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, required=True)
    parser.add_argument('--load_model', '-load', type=str, required=False)

    args = parser.parse_args()
    
    config = make_config(args.config_file)

    run_training_loop(config, args.load_model)

if __name__ == "__main__":
    main()