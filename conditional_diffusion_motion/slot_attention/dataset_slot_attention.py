import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PARTNET(Dataset):
    def __init__(self, split="train", image_dir: Path=None):
        super(PARTNET, self).__init__()

        assert split in ["train", "val", "test"]
        self.split = split
        self.image_dir = image_dir
        self.img_transform = transforms.Compose([transforms.ToTensor()])
        # Only include .png files
        self.files = [f for f in os.listdir(self.image_dir) if f.endswith(".png")]
        self.files.sort()

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.files[index])
        image = Image.open(img_path).convert("RGB")
        image = image.resize((128, 128))
        image = self.img_transform(image)
        sample = {"image": image}

        return sample

    def __len__(self):
        return len(self.files)
