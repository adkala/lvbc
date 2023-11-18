import scripting_utils
import utils

import argparse
import yaml
import os
import torch
import time

SAVE_ITER = 50

def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs |= yaml.load(f, Loader=yaml.SafeLoader)

    config = scripting_utils.get_make_model_config(config_kwargs['model'])(**config_kwargs)
    config.update(scripting_utils.get_make_dataset_config(config_kwargs['dataset'])(**config_kwargs))

    config['name'] = os.path.splitext(os.path.basename(config_file))[0] + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')

    if torch.cuda.is_available():
        config['device'] = torch.device('cuda')
        print(f'using cuda device {torch.cuda.current_device()}')
    else:
        config['device'] = torch.device('cpu')
        print(f'using cpu')

    config['model'].to(config['device'])

    return config

def run_training_loop(config, load_model=None):
    # create folder
    if not load_model:
        if not os.path.exists(f'models/{config["name"]}'):
            os.makedirs(f'models/{config["name"]}')

        loss_history = []
        ind_loss_history = []
        epochs = 0
    else:
        torch_dict = torch.load(load_model, map_location=torch.device('cpu'))

        config['model'].load_state_dict(torch_dict['model_state_dict'])
        config['optimizer'].load_state_dict(torch_dict['optimizer_state_dict'])

        loss_history = torch_dict['train_loss_history']
        ind_loss_history = torch_dict['train_ind_loss_history']
        epochs = torch_dict['epoch']

        config['name'] = torch_dict['name']

    for e in range(epochs + 1, config['epochs'] + epochs + 1):
        print(f'Starting epoch {e} / {epochs + config["epochs"]}')
        #loss, ind_loss = utils.lstm_training_loop(config)
        loss, ind_loss = utils.encdec_training_loop(config)
        loss_history.append(loss)
        ind_loss_history.append(ind_loss)
        if e % SAVE_ITER == 0:
            torch.save({
                'name': config['name'],
                'model_state_dict': config['model'].state_dict(),
                'epoch': e,
                'optimizer_state_dict': config['optimizer'].state_dict(),
                'train_loss_history': loss_history,
                'train_ind_loss_history': ind_loss_history,
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