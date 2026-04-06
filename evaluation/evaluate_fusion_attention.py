import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.dataset import EuroSATDataset
from utils.augment import get_val_transforms
from models.fusion_attention_model import FusionAttentionNet

# --------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

DATA_DIR = "data/eurosat_split/test"
WEIGHTS_DIR = "outputs/weights"
OUTPUT_DIR = "outputs/metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    times = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(DEVICE)
            labels = labels.numpy()

            start = time.time()
            outputs = model(imgs)
            end = time.time()

            times.append((end - start) / imgs.size(0))

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )

    return acc, prec, rec, f1, np.mean(times)


def main():
    print("\n📊 Evaluating Hybrid Fusion + Self-Attention Model\n")

    dataset = EuroSATDataset(DATA_DIR, transform=get_val_transforms())
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False)

    model = FusionAttentionNet(num_classes=10)
    weight_path = os.path.join(WEIGHTS_DIR, "fusion_attention_lulc.pth")

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Fusion-Attention weights not found at {weight_path}"
        )

    model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
    model.to(DEVICE)

    acc, prec, rec, f1, time_per_img = evaluate_model(model, dataloader)

    report = f"""
Hybrid Fusion + Self-Attention Evaluation Results (EuroSAT)

Accuracy  : {acc * 100:.2f} %
Precision : {prec * 100:.2f} %
Recall    : {rec * 100:.2f} %
F1-score  : {f1 * 100:.2f} %
Inference Time : {time_per_img * 1000:.2f} ms/image
"""

    print(report)

    output_file = os.path.join(
        OUTPUT_DIR, "fusion_attention_metrics.txt"
    )
    with open(output_file, "w") as f:
        f.write(report)

    print(f"✅ Metrics saved to {output_file}")


if __name__ == "__main__":
    main()
