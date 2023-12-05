from run import make_config

import argparse
import models
import torch


def run_training_loop(config, load_model):
    torch_dict = torch.load(load_model, map_location=config["device"])

    config["model"].load_state_dict(torch_dict["model_state_dict"], strict=False)

    loss_history = torch_dict["train_loss_history"]
    epochs = torch_dict["epoch"]

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

    run_training_loop(config, args.load_model)


if __name__ == "__main__":
    main()
