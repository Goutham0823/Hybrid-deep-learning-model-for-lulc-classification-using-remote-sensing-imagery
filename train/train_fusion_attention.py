import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import EuroSATDataset
from utils.augment import get_train_transforms, get_val_transforms
from models.fusion_attention_model import FusionAttentionNet

# --------------------------------------------------
# CONFIG (SAFE & CONTROLLED)
# --------------------------------------------------
TRAIN_DIR = "data/eurosat_split/train"
VAL_DIR   = "data/eurosat_split/test"

BATCH_SIZE = 32
EPOCHS = 25              # ↑ more epochs
LR = 3e-4                # ↓ safer LR
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
val_dataset = EuroSATDataset(
    VAL_DIR, transform=get_val_transforms()
)

train_loader = DataLoader(
    train_dataset, BATCH_SIZE,
    shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, BATCH_SIZE,
    shuffle=False, num_workers=0
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
model = FusionAttentionNet(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

best_val_acc = 0.0

# --------------------------------------------------
# TRAINING LOOP
# --------------------------------------------------
for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0
    train_loss = 0.0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ---------------- VALIDATION ----------------
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Acc: {train_acc:.2f}% | "
        f"Val Acc: {val_acc:.2f}%"
    )

    # ---------------- SAVE BEST ONLY ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(
            model.state_dict(),
            "outputs/weights/fusion_attention_lulc.pth"
        )
        print(f"✅ Best model saved (Val Acc: {best_val_acc:.2f}%)")

print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
