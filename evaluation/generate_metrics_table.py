import re
from tabulate import tabulate

METRIC_FILES = {
    "DenseNet-64": "outputs/metrics/densenet64_metrics.txt",
    "IBNR-65": "outputs/metrics/ibnr65_metrics.txt",
    "IBNR-65 + DenseNet-64 (Fusion)": "outputs/metrics/fusion_metrics.txt",
    "Hybrid Fusion + Self-Attention (Proposed)": "outputs/metrics/fusion_attention_metrics.txt"
}

def extract_metrics(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    acc = re.search(r"Accuracy\s*:\s*([\d.]+)", text).group(1)
    prec = re.search(r"Precision\s*:\s*([\d.]+)", text).group(1)
    rec = re.search(r"Recall\s*:\s*([\d.]+)", text).group(1)
    f1 = re.search(r"F1-score\s*:\s*([\d.]+)", text).group(1)
    time = re.search(r"Inference Time\s*:\s*([\d.]+)", text).group(1)

    return prec, rec, f1, acc, time


table = []

for model, path in METRIC_FILES.items():
    metrics = extract_metrics(path)
    table.append([model, *metrics])

headers = [
    "Classifier",
    "Precision (%)",
    "Recall (%)",
    "F1-Score (%)",
    "Accuracy (%)",
    "Inference Time (ms/img)"
]

print("\n📊 FINAL CLASSIFIER-WISE PERFORMANCE TABLE\n")
print(tabulate(table, headers=headers, tablefmt="grid"))
