from torch.utils.data import Dataset
import os
from PIL import Image

class classification_dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')] # get data classes
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)} # class to index mapping

        self.image_paths = []
        self.labels = []

        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for dirpath, _, filenames in os.walk(class_dir):
                for filename in filenames:
                    img_path = os.path.join(dirpath, filename)
                    if img_path.endswith(('jpg', 'jpeg', 'png')):
                        self.image_paths.append(img_path)
                        self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label