import os
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random

from models.fusion_attention_model import FusionAttentionNet
from models.cropland_model import CroplandSuitabilityModel
from models.croptype_model import CropTypeModel

# --------------------------------------------------
# Deterministic inference (same image -> same result)
# --------------------------------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- IMAGE TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- LULC CLASSES ----------------
LULC_CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]

# ---------------- INDIAN CROP CLASSES ----------------
CROP_CLASSES = [
    "Rice",
    "Wheat",
    "Maize",
    "Cotton",
    "Sugarcane",
    "Millets",
    "Groundnut",
    "Pulses",
    "Mustard",
    "Tea",
    "Coffee"
]

# ---------------- CROPLAND RULES ----------------
STRONG_CROPLAND = ["AnnualCrop", "PermanentCrop"]
PARTIAL_CROPLAND = ["Pasture"]
NON_CROPLAND = [
    "Forest", "HerbaceousVegetation", "Residential",
    "Industrial", "Highway", "River", "SeaLake"
]

# ---------------- MODEL LOADER ----------------
def load_model(model, path):
    if not os.path.exists(path):
        raise RuntimeError(f"Missing model file: {os.path.abspath(path)}")

    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# ---------------- LOAD MODELS ----------------
lulc_model = load_model(
    FusionAttentionNet(num_classes=10),
    "outputs/weights/fusion_attention_lulc.pth"
)

cropland_model = load_model(
    CroplandSuitabilityModel(),
    "outputs/weights/cropland_suitability.pth"
)

crop_type_model = load_model(
    CropTypeModel(),
    "outputs/weights/crop_type.pth"
)


# ---------------- PREDICT FUNCTION ----------------
def predict(image: Image.Image):

    img = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # -------- LULC PREDICTION --------
        lulc_logits = lulc_model(img)
        lulc_probs = F.softmax(lulc_logits, dim=1)

        lulc_conf, lulc_idx = torch.max(lulc_probs, dim=1)

        lulc_label = LULC_CLASSES[lulc_idx.item()]
        lulc_conf = float(lulc_conf.item() * 100)

        # -------- NON CROPLAND --------
        if lulc_label in NON_CROPLAND:
            return {
                "lulc": (lulc_label, lulc_conf),
                "cropland": ("Not suitable for cropland", 100.0),
                "crop": (None, None),
                "top_crops": []
            }

        # -------- CROPLAND SUITABILITY --------
        cropland_logits = cropland_model(img)
        cropland_probs = F.softmax(cropland_logits, dim=1)

        cropland_conf, cropland_idx = torch.max(cropland_probs, dim=1)
        cropland_conf = float(cropland_conf.item() * 100)

        cropland_status = (
            "Suitable for cropland"
            if lulc_label in STRONG_CROPLAND
            else "Partially suitable for cropland"
        )

        # -------- CROP TYPE PREDICTION (TOP 3) --------
        crop_logits = crop_type_model(img)
        crop_probs = F.softmax(crop_logits, dim=1)

        top_conf, top_idx = torch.topk(crop_probs, k=3)

        top_conf = top_conf[0]
        top_idx = top_idx[0]

        crop_names = [CROP_CLASSES[i] for i in top_idx.tolist()]
        crop_confs = [float(c.item() * 100) for c in top_conf]

    return {
        "lulc": (lulc_label, lulc_conf),
        "cropland": (cropland_status, cropland_conf),
        "crop": (crop_names[0], crop_confs[0]),
        "top_crops": list(zip(crop_names, crop_confs))
    }