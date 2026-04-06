import os
import shutil
import random

SOURCE_DIR = "data/eurosat"
TARGET_DIR = "data/eurosat_split"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

def split_dataset():
    classes = [
        d for d in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, d))
    ]

    for split in ["train", "test"]:
        for cls in classes:
            os.makedirs(os.path.join(TARGET_DIR, split, cls), exist_ok=True)

    for cls in classes:
        cls_path = os.path.join(SOURCE_DIR, cls)
        images = [
            img for img in os.listdir(cls_path)
            if img.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        random.shuffle(images)
        split_idx = int(len(images) * TRAIN_RATIO)

        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(TARGET_DIR, "train", cls, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(TARGET_DIR, "test", cls, img)
            )

    print(" Dataset split completed!")
    print("Train/Test folders created in data/eurosat_split")

if __name__ == "__main__":
    split_dataset()
