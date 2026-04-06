import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import EuroSATDataset
from utils.augment import get_train_transforms, get_val_transforms
from models.ibnr65 import IBNR65

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TRAIN_DIR = "data/eurosat_split/train"
TEST_DIR  = "data/eurosat_split/test"

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
NUM_CLASSES = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

os.makedirs("outputs/weights", exist_ok=True)

# --------------------------------------------------
# DATA
# --------------------------------------------------
train_dataset = EuroSATDataset(
    TRAIN_DIR, transform=get_train_transforms()
)
test_dataset = EuroSATDataset(
    TEST_DIR, transform=get_val_transforms()
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = IBNR65(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )

# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------
save_path = "outputs/weights/ibnr65_lulc.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ IBNR-65 model saved at {save_path}")
