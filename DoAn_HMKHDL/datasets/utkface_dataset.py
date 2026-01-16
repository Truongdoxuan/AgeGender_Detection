import os
import json
import cv2
import torch
from torch.utils.data import Dataset


class UTKFaceDataset(Dataset):
    """
    Custom Dataset for UTKFace
    Filename format: age_gender_race_date.jpg
    Example: 25_0_0_20170116.jpg
    """

    def __init__(self, image_dir, stats_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load age normalization stats
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.mean_age = stats["mean_age"]
        self.std_age = stats["std_age"]

        # Load image file list
        self.image_files = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ]

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        try:
            age, gender, *_ = img_name.split("_")
            age = float(age)
            gender = float(gender)  # 0: male, 1: female
        except Exception:
            raise ValueError(f"Invalid filename format: {img_name}")

        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        age_norm = (age - self.mean_age) / self.std_age

        return (
            image,                            
            torch.tensor(age_norm, dtype=torch.float32),
            torch.tensor(gender, dtype=torch.float32)
        )
