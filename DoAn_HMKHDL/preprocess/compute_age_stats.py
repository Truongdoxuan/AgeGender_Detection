import os
import numpy as np
import json

DATA_DIR = "../data/UTKFace_cropped"

ages = []
for name in os.listdir(DATA_DIR):
    if not name.endswith(".jpg"):
        continue
    age = int(name.split("_")[0])
    ages.append(age)

mean_age = float(np.mean(ages))
std_age = float(np.std(ages))

print("Mean age:", mean_age)
print("Std age:", std_age)

with open("../data/stats.json", "w") as f:
    json.dump(
        {"mean_age": mean_age, "std_age": std_age},
        f, indent=4
    )
