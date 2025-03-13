import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def create_overlay(img, mask, alpha=0.5, color=(0, 255, 0)):
    # Convert to 0-1 range
    color = np.array(color) / 255
    mask = (mask > 0).astype(np.float32)

    # Convert to RGB with given color
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    mask = mask * color

    # Overlay
    overlay = np.clip(img * (1 - alpha) + mask * alpha, 0, 1)
    return overlay


def generate_plot(img, mask, save_path, alpha, highlight_color):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Predicted mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    # Overlay
    overlay = create_overlay(img, mask, alpha, highlight_color)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    # Adjust layout and add extra space for titles
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save plot
    plt.savefig(save_path)
    plt.close(fig)


class Predictor:
    def __init__(
        self, model, dataset, predict_args, weights_path="", save_path=""
    ):
        self.model = model
        self.dataset = dataset

        self.threshold = predict_args["threshold"]
        self.alpha = predict_args["alpha"]
        self.highlight_color = predict_args["highlight_color"]

        self.weights_path = weights_path
        self.save_path = save_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"Using device: {self.device}")

    def predict(self):
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path))
        else:
            print("No pretrained weights provided!")

        self.model.to(self.device)  # Move model to device
        self.model.eval()  # Set model to evaluation mode

        dataloader = DataLoader(
            self.dataset, batch_size=1, pin_memory=False, shuffle=False
        )

        with torch.no_grad():
            for i, img in enumerate(tqdm(dataloader)):
                img = img.float().to(self.device)

                # Perform inference
                pred = self.model(img)

                # Convert to numpy array of probabilities
                pred = pred.squeeze()
                pred = torch.sigmoid(pred)
                pred = pred.cpu().detach().numpy()

                # Values greater than threshold are set to 255 which is white
                pred = (pred > self.threshold) * 255
                pred = pred.astype(np.uint8)

                # Convert to numpy array with correct shape
                img = img.squeeze()
                img = img.cpu().detach().numpy()
                img = img.transpose(1, 2, 0)

                # Generate plot and save to file
                save_path = os.path.join(self.save_path, f"plot{i}.png")
                generate_plot(
                    img, pred, save_path, self.alpha, self.highlight_color
                )
