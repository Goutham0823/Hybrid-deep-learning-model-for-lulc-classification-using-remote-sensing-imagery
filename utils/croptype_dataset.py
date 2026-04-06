from utils.dataset import EuroSATDataset

CROP_CLASSES = ["AnnualCrop", "PermanentCrop", "Pasture"]

class CropTypeDataset(EuroSATDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)

        self.crop_images = []
        self.crop_labels = []

        for img_path, label in zip(self.images, self.labels):
            class_name = img_path.split("\\")[-2]
            if class_name in CROP_CLASSES:
                self.crop_images.append(img_path)
                self.crop_labels.append(CROP_CLASSES.index(class_name))

        self.images = self.crop_images
        self.labels = self.crop_labels
