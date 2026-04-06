import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.cropland_dataset import CroplandDataset
from utils.augment import get_train_transforms, get_val_transforms
from models.cropland_model import CroplandSuitabilityModel


def main():
    TRAIN_DIR = "data/eurosat_split/train"
    TEST_DIR  = "data/eurosat_split/test"

    BATCH_SIZE = 32
    EPOCHS = 2          # keep small for now
    LR = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = CroplandDataset(TRAIN_DIR, transform=get_train_transforms())
    test_dataset  = CroplandDataset(TEST_DIR, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader  = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

    model = CroplandSuitabilityModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # -------- TRAIN LOOP --------
    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Acc: {acc:.2f}%")

    # -------- SAVE (ABSOLUTE PATH) --------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weights_dir = os.path.join(base_dir, "outputs", "weights")
    os.makedirs(weights_dir, exist_ok=True)

    save_path = os.path.join(weights_dir, "cropland_suitability.pth")
    torch.save(model.state_dict(), save_path)

    print("\n✅ CROPLAND MODEL SAVED SUCCESSFULLY")
    print("📍 Location:", save_path)
    print("📦 File exists:", os.path.exists(save_path))


if __name__ == "__main__":
    main()
