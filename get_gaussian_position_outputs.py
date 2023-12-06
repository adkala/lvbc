from run import make_config
from torch import nn

import argparse
import models
import torch
import utils
import os
import numpy as np

BATCH_SIZE = 256


def run_training_loop(config, load_model):
    if not os.path.exists(f'models/{config["name"]}'):
        os.makedirs(f'models/{config["name"]}')

    loss_history = []
    epochs = 0

    torch_dict = torch.load(load_model, map_location=config["device"])

    config["model"].load_state_dict(torch_dict["model_state_dict"], strict=False)

    for e in range(epochs + 1, config["epochs"]):
        print(f'Starting epoch {e} / {config["epochs"]}')
        loss = config["training_loop"](config["train_datasets"])
        loss_history.append(loss)
        if e % config["save_iter"] == 0:
            torch.save(
                {
                    "name": config["name"],
                    "epoch": e,
                    "model_state_dict": config["model"].state_dict(),
                    "optimizer_state_dict": config["optimizer"].state_dict(),
                    "train_loss_history": loss_history,
                },
                f'models/{config["name"]}/{config["name"]}_e{e}.p',
            )
        print(f"Epoch {e} loss: {loss} \n")

        # check variance
        i = np.random.randint(len(config['train_datasets']))
        j = np.random.randint(len(config['train_datasets'][i]))
        x, _ = config['train_datasets'][i][j]
        y_pred = config['model'](torch.tensor(x).float())
        print(y_pred.shape)
        var = y_pred.detach().numpy()[:, 3:]
        print(f'variance ({i}, {j}): {var.mean(axis=0)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)
    parser.add_argument("--load_model", "-load", type=str, required=False)

    args = parser.parse_args()

    config = make_config(args.config_file)

    # change model
    config["model"] = models.HorizonLSTMWithVariance()
    config["model"].to(config["device"])

    # freeze layers
    for name, module in config["model"].named_children():
        if "var" not in name:
            print(f"freezing {name}")
            for param in module.parameters():
                param.requires_grad = False
        else:
            print(f"not freezing {name}")

    config["optimizer"] = torch.optim.Adam(
        config["model"].parameters(), lr=config["optimizer"].param_groups[0]["lr"]
    )

    # criterion
    gnl = nn.GaussianNLLLoss()
    criterion = lambda y_pred, y: gnl(
        y_pred[:, :, : y.shape[-1]],
        y,
        y_pred[:, :, y.shape[-1] :],
    )  # pred, target, var (relu to keep +)

    # training loop
    config["training_loop"] = lambda datasets: utils.cont_training_loop(
        config["model"],
        config["optimizer"],
        criterion,
        datasets,
        config["device"],
        batch_size=BATCH_SIZE,
        comp_error=False,
    )

    run_training_loop(config, args.load_model)


if __name__ == "__main__":
    main()
