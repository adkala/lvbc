from tqdm import tqdm

import scripting_utils
import utils

import argparse
import yaml
import os
import torch
import time
import pickle

SAVE_ITER = 10

def make_config(config_file: str) -> dict:
    config_kwargs = {}
    with open(config_file, "r") as f:
        config_kwargs |= yaml.load(f, Loader=yaml.SafeLoader)

    config = scripting_utils.make_config(**config_kwargs)

    config['name'] = os.path.splitext(os.path.basename(config_file))[0] + '_' + time.strftime('%d-%m-%Y_%H-%M-%S')

    if torch.cuda.is_available():
        config['device'] = torch.device('cuda')
        print(f'using cuda device {torch.cuda.current_device()}')
    else:
        config['device'] = torch.device('cpu')
        print(f'using cpu')

    return config

def run_training_loop(config):
    # create folders
    os.makedirs(f'logs/{config["name"]}')
    os.makedirs(f'models/{config["name"]}')

    pbar = tqdm(range(config['epochs']), desc="Epoch Loss: N/A")
    for e in pbar:
        loss, ind_loss = utils.training_loop(config)
        if e % SAVE_ITER == 0:
            torch.save(config['model'], f'models/{config["name"]}/{config["name"]}_e{e}.p')
            with open(f'logs/{config["name"]}/{config["name"]}_e{e}.p', 'wb') as f:
                pickle.dump((loss, ind_loss), f)
        pbar.set_description(f'Epoch Loss: {loss}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-cfg', type=str, required=True)

    args = parser.parse_args()

    config = make_config(args.config_file)

    run_training_loop(config)

if __name__ == "__main__":
    main()