import os
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

INPUT_DIR = "../data/UTKFace"
OUTPUT_DIR = "../data/UTKFace_cropped"

os.makedirs(OUTPUT_DIR, exist_ok=True)

mtcnn = MTCNN(
    image_size=224,
    margin=20,
    keep_all=False,
    post_process=True
)

for img_name in tqdm(os.listdir(INPUT_DIR)):
    if not img_name.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(INPUT_DIR, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
        save_path = os.path.join(OUTPUT_DIR, img_name)

        mtcnn(img, save_path=save_path)
    except Exception as e:
        print(f"Skip {img_name}: {e}")
