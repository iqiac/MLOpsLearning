import argparse
import json

import torch
from dataset import SegmentationDataset
from model import UNet
from predictor import Predictor
from torchinfo import summary
from trainer import Trainer


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
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
    modes = ["train", "predict", "all"]
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        default="",
        choices=modes,
        help="Choose mode train or predict",
    )

    args = parser.parse_args()
    config = load_json(args.config)

    state_args = config["state"]
    model_args = config["model"]
    train_args = config["trainer"]
    predict_args = config["predictor"]
    data_args = config["data"]

    set_seed(state_args["seed"])
    mode = args.mode if args.mode else state_args["mode"]
    if mode not in modes:
        print(
            f"""
            Invalid mode: {mode}.
            Available modes: {modes}.
            Defaulting to all.
            """
        )
        mode = "all"

    model = UNet(
        model_args["in_channels"],
        model_args["num_classes"],
        model_args["dropout"],
    )

    print("Model summary:")
    summary(
        model,
        input_size=(
            train_args["batch_size"],
            model_args["in_channels"],
            256,
            256,
        ),
    )

    # Training mode
    if mode == "train" or mode == "all":
        print("Starting training...")

        dataset = SegmentationDataset(
            data_args["img_path"],
            data_args["mask_path"],
            data_args["train_resize"],
        )

        trainer = Trainer(
            model,
            dataset,
            train_args,
            model_args["weights_path"],
            model_args["save_path"],
        )

        trainer.train()

    # Inference mode
    if mode == "predict" or mode == "all":
        print("Starting inference...")

        dataset = SegmentationDataset(
            data_args["predict_src"], resize=data_args["predict_resize"]
        )

        predictor = Predictor(
            model,
            dataset,
            predict_args,
            model_args["weights_path"],
            data_args["predict_dst"],
        )

        predictor.predict()


if __name__ == "__main__":
    main()
