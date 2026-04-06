from utils.dataset import EuroSATDataset

CROPLAND_CLASSES = {
    "AnnualCrop",
    "PermanentCrop",
    "Pasture"
}

class CroplandDataset(EuroSATDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        # Map original labels to binary labels
        self.binary_labels = []
        for path in self.images:
            class_name = path.split("\\")[-2]
            if class_name in CROPLAND_CLASSES:
                self.binary_labels.append(1)  # Cropland
            else:
                self.binary_labels.append(0)  # Non-cropland

    def __getitem__(self, idx):
        image, _ = super().__getitem__(idx)
        label = self.binary_labels[idx]
        return image, label
