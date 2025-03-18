import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import v2


class SegmentationDataset(Dataset):
    def __init__(
        self, img_path, label_path=None, augment=False, resize=(512, 512)
    ):
        self.images = sorted(
            [os.path.join(img_path, name) for name in os.listdir(img_path)]
        )
        if label_path:
            self.masks = sorted(
                [
                    os.path.join(label_path, name)
                    for name in os.listdir(label_path)
                ]
            )
        else:
            self.masks = None

        self.transform = v2.Compose(
            [
                v2.RandomHorizontalFlip(0.5) if augment else v2.Identity(),
                v2.RandomVerticalFlip(0.2) if augment else v2.Identity(),
                v2.Resize(resize),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        if self.masks is None:
            return self.transform(img)
        mask = Image.open(self.masks[index]).convert("L")

        img, mask = self.transform(img, mask)
        return img, mask

    def __len__(self):
        return len(self.images)
