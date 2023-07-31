import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data_file = data_file
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]["image_path"]
        image = Image.open(image_path).convert("RGB")

        # Load other information from the JSON file, e.g., labels and bounding boxes
        label = self.data[idx]["label"]
        bounding_box = self.data[idx]["bounding_box"]

        if self.transform:
            image = self.transform(image)
            # Apply transformations to bounding boxes if needed

        # Return a dictionary containing the image, label, and bounding box
        sample = {"image": image, "label": label, "bounding_box": bounding_box}
        return sample



