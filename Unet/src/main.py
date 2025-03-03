import argparse
import json
import random

import numpy as np
import torch
from dataset import SegmentationDataset
from model import UNet
from trainer import Trainer


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="../config/config.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="predict",
        choices=["train", "predict"],
        help="Choose mode train or predict",
    )

    args = parser.parse_args()
    mode = args.mode
    config = load_json(args.config)

    state_args = config["state"]
    model_args = config["model"]
    train_args = config["train"]
    data_args = config["data"]

    set_seed(state_args["seed"])

    model = UNet(model_args["in_channels"], model_args["num_classes"])
    dataset = SegmentationDataset(data_args["img_path"], data_args["mask_path"])

    trainer = Trainer(
        model,
        dataset,
        train_args,
        model_args["weights_path"],
        model_args["save_path"],
    )
    if mode == "train":
        trainer.train()
    elif mode == "predict":
        trainer.predict()


if __name__ == "__main__":
    main()
