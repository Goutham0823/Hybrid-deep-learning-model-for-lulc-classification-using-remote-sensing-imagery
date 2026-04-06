import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset import EuroSATDataset
from utils.augment import get_train_transforms, get_val_transforms
from models.ibnr65 import IBNR65

# ---------------- CONFIG ----------------
TRAIN_DIR = "data/eurosat_split/train"
TEST_DIR  = "data/eurosat_split/test"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 10
# ----------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------- DATASETS --------
train_dataset = EuroSATDataset(TRAIN_DIR, transform=get_train_transforms())
test_dataset  = EuroSATDataset(TEST_DIR,  transform=get_val_transforms())

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True
)

# -------- MODEL --------
model = IBNR65(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------- TRAIN LOOP --------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # -------- TEST LOOP --------
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

    test_acc = 100 * test_correct / test_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Loss: {train_loss:.4f} "
          f"Train Acc: {train_acc:.2f}% "
          f"Test Acc: {test_acc:.2f}%")

# -------- SAVE MODEL --------
torch.save(model.state_dict(), "outputs/weights/ibnr65_lulc.pth")
print("✅ Model saved!")
