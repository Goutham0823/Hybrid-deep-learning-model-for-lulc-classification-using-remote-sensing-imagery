import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.croptype_dataset import CropTypeDataset
from utils.augment import get_train_transforms, get_val_transforms
from models.croptype_model import CropTypeModel

TRAIN_DIR = "data/eurosat_split/train"
TEST_DIR  = "data/eurosat_split/test"

BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

train_dataset = CropTypeDataset(TRAIN_DIR, transform=get_train_transforms())
test_dataset  = CropTypeDataset(TEST_DIR, transform=get_val_transforms())

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=0)

model = CropTypeModel().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    correct, total = 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(out, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            _, preds = torch.max(out, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = 100 * correct / total

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Acc: {train_acc:.2f}% "
          f"Test Acc: {test_acc:.2f}%")

torch.save(model.state_dict(), "outputs/weights/crop_type.pth")
print("✅ Crop type model saved")
