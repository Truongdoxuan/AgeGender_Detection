import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# =====================
# ADD ROOT PROJECT PATH
# =====================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets.utkface_dataset import UTKFaceDataset
from models.age_gender_net import AgeGenderNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =====================
# SEED (REPRODUCIBILITY)
# =====================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything()

# =====================
# CONFIG
# =====================
DATA_DIR = "../data/UTKFace_cropped"
STATS_PATH = "../data/stats.json"
BATCH_SIZE = 8
EPOCHS = 25
LR = 1e-4
LAMBDA_AGE = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = "checkpoints"
PLOT_DIR = "plots"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =====================
# LOAD AGE STATS
# =====================
with open(STATS_PATH) as f:
    stats = json.load(f)

MEAN_AGE = stats["mean_age"]
STD_AGE = stats["std_age"]

# =====================
# TRANSFORMS
# =====================
train_tfms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.5
    ),
    A.Normalize(),
    ToTensorV2()
])

val_tfms = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# =====================
# DATASET & DATALOADER
# =====================
full_dataset = UTKFaceDataset(DATA_DIR, STATS_PATH, transform=train_tfms)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# override transform for validation
val_ds.dataset.transform = val_tfms

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# =====================
# MODEL
# =====================
model = AgeGenderNet().to(DEVICE)

# =====================
# LOSS & OPTIMIZER
# =====================
gender_criterion = nn.BCEWithLogitsLoss()
age_criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# =====================
# METRICS STORAGE
# =====================
history = {
    "train_loss": [],
    "val_loss": [],
    "val_mae_real": [],
    "val_gender_acc": []
}

best_mae = float("inf")

# =====================
# TRAIN LOOP
# =====================
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, ages, genders in train_loader:
        imgs = imgs.to(DEVICE)
        ages = ages.to(DEVICE).unsqueeze(1)
        genders = genders.to(DEVICE).unsqueeze(1)

        gender_logits, age_preds = model(imgs)

        gender_loss = gender_criterion(gender_logits, genders)
        age_loss = age_criterion(age_preds, ages)

        loss = gender_loss + LAMBDA_AGE * age_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)
    history["train_loss"].append(train_loss)

    # =====================
    # VALIDATION
    # =====================
    model.eval()
    val_loss = 0.0
    mae_real = 0.0
    correct_gender = 0
    total = 0

    with torch.no_grad():
        for imgs, ages, genders in val_loader:
            imgs = imgs.to(DEVICE)
            ages = ages.to(DEVICE).unsqueeze(1)
            genders = genders.to(DEVICE).unsqueeze(1)

            gender_logits, age_preds = model(imgs)

            gender_loss = gender_criterion(gender_logits, genders)
            age_loss = age_criterion(age_preds, ages)
            loss = gender_loss + LAMBDA_AGE * age_loss

            val_loss += loss.item()

            # ===== REAL AGE MAE =====
            age_preds_real = age_preds * STD_AGE + MEAN_AGE
            ages_real = ages * STD_AGE + MEAN_AGE
            mae_real += torch.abs(age_preds_real - ages_real).mean().item()

            # ===== GENDER ACC =====
            preds = (torch.sigmoid(gender_logits) > 0.5)
            correct_gender += (preds == genders.bool()).sum().item()
            total += genders.size(0)

    val_loss /= len(val_loader)
    mae_real /= len(val_loader)
    gender_acc = correct_gender / total

    history["val_loss"].append(val_loss)
    history["val_mae_real"].append(mae_real)
    history["val_gender_acc"].append(gender_acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"MAE (years): {mae_real:.2f} | "
        f"Gender Acc: {gender_acc:.4f}"
    )

    # =====================
    # SAVE BEST MODEL
    # =====================
    if mae_real < best_mae:
        best_mae = mae_real
        torch.save(
            model.state_dict(),
            os.path.join(CHECKPOINT_DIR, "best_model.pth")
        )

# =====================
# PLOT & SAVE FIGURES
# =====================

# Loss curve
plt.figure()
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "loss_curve.png"), dpi=300)
plt.close()

# MAE curve (real age)
plt.figure()
plt.plot(history["val_mae_real"])
plt.xlabel("Epoch")
plt.ylabel("MAE (Years)")
plt.title("Validation MAE (Real Age)")
plt.savefig(os.path.join(PLOT_DIR, "mae_curve.png"), dpi=300)
plt.close()

# Gender accuracy curve
plt.figure()
plt.plot(history["val_gender_acc"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Gender Accuracy")
plt.savefig(os.path.join(PLOT_DIR, "gender_acc_curve.png"), dpi=300)
plt.close()
