import matplotlib.pyplot as plt
import mlflow
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def plot(epochs, train_losses, train_dcs, val_losses, val_dcs):
    x_axis = range(1, epochs + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x_axis, train_losses, label="Training Loss")
    plt.plot(x_axis, val_losses, label="Validation Loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x_axis, train_dcs, label="Training DICE Coefficient")
    plt.plot(x_axis, val_dcs, label="Validation DICE Coefficient")
    plt.title("DICE Coefficent over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("DICE")
    plt.grid()
    plt.tight_layout()
    plt.legend()

    plt.show()


def dice_coefficient(mask_true, mask_pred):
    mask_pred_clone = mask_pred.clone()
    mask_pred_clone[mask_pred_clone > 0] = 1
    mask_pred_clone[mask_pred_clone < 0] = 0

    intersection = abs(torch.sum(mask_pred_clone * mask_true))
    union = abs(torch.sum(mask_pred_clone) + torch.sum(mask_true))

    eps = 1e-7
    return (2.0 * intersection + eps) / (union + eps)


class Trainer:
    def __init__(
        self, model, dataset, train_args, weights_path="", save_path=""
    ):
        self.model = model
        self.dataset = dataset

        self.mlflow_uri = train_args["mlflow_uri"]
        self.use_pretrained = train_args["use_pretrained"]
        self.train_val_split = train_args["train_val_split"]
        self.epochs = train_args["epochs"]
        self.batch_size = train_args["batch_size"]
        self.lr = train_args["lr"]

        self.weights_path = weights_path
        self.save_path = save_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            torch.cuda.empty_cache()
        print(f"Using device: {self.device}")

    def train(self):
        # Split data into train, val and test set
        train_set, val_set = random_split(self.dataset, self.train_val_split)
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            pin_memory=False,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, pin_memory=False, shuffle=True
        )

        # Define loss function and optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Use pretrained weights, if available
        if self.use_pretrained and self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path))
        self.model.to(self.device)  # Move model to device

        train_losses, val_losses = [], []
        train_dcs, val_dcs = [], []

        # Initialize mlflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("Unet")
        mlflow.pytorch.autolog()

        # Train model
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                }
            )
            for epoch in tqdm(range(self.epochs)):
                self.model.train()  # Set model to training mode

                train_loss, val_loss = 0, 0
                train_dc, val_dc = 0, 0

                for img, mask in tqdm(train_loader, position=0, leave=True):
                    img = img.float().to(self.device)
                    mask = mask.float().to(self.device)

                    # Perform forward pass and calculate training loss and dc
                    pred = self.model(img)
                    loss = loss_fn(pred, mask)
                    dc = dice_coefficient(mask, pred)

                    # Add loss and dc to running total
                    train_loss += loss.item()
                    train_dc += dc.item()

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Perform backward pass and update weights
                    loss.backward()
                    optimizer.step()

                # Add average loss and dc to list
                train_losses.append(train_loss / len(train_loader))
                train_dcs.append(train_dc / len(train_loader))

                # Evaluate model on validation set
                self.model.eval()  # Set model to evaluation mode
                with torch.no_grad():
                    for img, mask in tqdm(val_loader, position=0, leave=True):
                        img = img.float().to(self.device)
                        mask = mask.float().to(self.device)

                        # Perform forward pass and calculate validation loss and dc
                        pred = self.model(img)
                        loss = loss_fn(pred, mask)
                        dc = dice_coefficient(mask, pred)

                        # Add loss and dc to running total
                        val_loss += loss.item()
                        val_dc += dc.item()

                # Add average loss and dc to list
                val_losses.append(val_loss / len(val_loader))
                val_dcs.append(val_dc / len(val_loader))

                print("-" * 30)
                print(
                    f"Training Loss Epoch {epoch + 1}: {train_losses[-1]:.5f}"
                )
                print(f"Training DC Epoch {epoch + 1}: {train_dcs[-1]:.5f}")
                print()
                print(
                    f"Validation Loss Epoch {epoch + 1}: {val_losses[-1]:.5f}"
                )
                print(f"Validation DC Epoch {epoch + 1}: {val_dcs[-1]:.5f}")
                print("-" * 30)

                # Log metrics to mlflow
                mlflow.log_metric("train_loss", train_losses[-1], epoch + 1)
                mlflow.log_metric("train_dc", train_dcs[-1], epoch + 1)
                mlflow.log_metric("val_loss", val_losses[-1], epoch + 1)
                mlflow.log_metric("val_dc", val_dcs[-1], epoch + 1)

        # Log model to mlflow
        mlflow.pytorch.log_model(self.model, "model")

        # Save model weights
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

        # Plot loss and DICE
        plot(self.epochs, train_losses, train_dcs, val_losses, val_dcs)
