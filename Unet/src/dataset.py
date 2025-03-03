import os

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class SegmentationDataset(Dataset):
    def __init__(self, img_path, label_path):
        self.images = sorted(
            [os.path.join(img_path, name) for name in os.listdir(img_path)]
        )
        self.masks = sorted(
            [os.path.join(label_path, name) for name in os.listdir(label_path)]
        )
        self.transform = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor()]
        )

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")

        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)
