import torch
import cv2
import numpy as np
from models.age_gender_net import AgeGenderNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== LOAD STATS =====
with open("data/stats.json", "r") as f:
    stats = json.load(f)

MEAN_AGE = stats["mean_age"]
STD_AGE = stats["std_age"]

# ===== TRANSFORM =====
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(),
    ToTensorV2()
])

# ===== LOAD MODEL =====
model = AgeGenderNet().to(DEVICE)
model.load_state_dict(
    torch.load("train/checkpoints/best_model.pth", map_location=DEVICE)
)
model.eval()


def predict_face(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    augmented = transform(image=img_rgb)
    img_tensor = augmented["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        gender_logit, age_norm = model(img_tensor)

    # ===== GENDER =====
    gender_prob = torch.sigmoid(gender_logit).item()
    gender = "Female" if gender_prob > 0.5 else "Male"


    # ===== AGE (DENORMALIZE) =====
    age_real = age_norm.item() * STD_AGE + MEAN_AGE

    # ===== HEURISTIC FIX =====
    if age_real < 10:
        age_real = np.clip(age_real, 1, 12)
    elif age_real < 25:
        age_real = np.clip(age_real, 12, 30)
    elif age_real < 40:
        age_real = np.clip(age_real, 25, 45)
    else:
        age_real = np.clip(age_real, 40, 80)

    age_real = int(round(age_real))

    return gender, age_real
