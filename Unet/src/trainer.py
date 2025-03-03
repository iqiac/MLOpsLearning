import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


def dice_coefficient(mask_true, mask_pred):
    mask_pred_clone = mask_pred.clone()
    mask_pred_clone[mask_pred_clone > 0] = 1
    mask_pred_clone[mask_pred_clone < 0] = 0

    intersection = abs(torch.sum(mask_true * mask_pred_clone))
    union = abs(torch.sum(mask_true)) + abs(torch.sum(mask_pred_clone))
    if union == 0:
        return 0
    return 2 * intersection / union


class Trainer:
    def __init__(
        self, model, dataset, train_args, weights_path="", save_path=""
    ):
        self.model = model
        self.dataset = dataset

        self.train_val_test_split = train_args["train_val_test_split"]
        self.epochs = train_args["epochs"]
        self.batch_size = train_args["batch_size"]
        self.lr = train_args["lr"]
        self.cross_validation = train_args["cross_validation"]

        self.weights_path = weights_path
        self.save_path = save_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

    def train(self):
        # Split data into train, val and test set
        train_set, val_set, test_set = random_split(
            self.dataset, self.train_val_test_split
        )
        train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=True
        )
        test_loader = DataLoader(
            test_set, batch_size=self.batch_size, shuffle=True
        )

        # Define loss function and optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        optimizer.zero_grad()
        loss_fn = torch.nn.BCEWithLogitsLoss()

        # Use pretrained weights, if available
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path))
        self.model.to(self.device)  # Move model to device

        train_losses, val_losses = [], []
        train_dcs, val_dcs = [], []

        # Train model
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

                # Zero the gradients
                optimizer.zero_grad()

                # Perform backward pass and update weights
                loss.backward()
                optimizer.step()

                # Add loss and dc to running total
                train_loss += loss.item()
                train_dc += dc.item()

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
            print(f"Training Loss Epoch {epoch + 1}: {train_losses[-1]:.3f}")
            print(f"Training DC Epoch {epoch + 1}: {train_dcs[-1]:.3f}")
            print()
            print(f"Validation Loss Epoch {epoch + 1}: {val_losses[-1]:.3f}")
            print(f"Validation DC Epoch {epoch + 1}: {val_dcs[-1]:.3f}")
            print("-" * 30)

        # Save model weights
        if self.save_path:
            torch.save(self.model.state_dict(), self.save_path)

    def predict(self):
        pass
