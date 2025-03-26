# libraries

import os

from torch.utils.data import Dataset
from config import CLASSES

import torch


class EuroSATDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the dataset folder containing subfolders (each representing a class).
            classes (list, optional): List of class names to use. If None, all classes in root_dir are used.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []

        # If classes are not provided, infer from directory structure
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES)}

        # Collect all image paths and labels
        for cls in CLASSES:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):  # Ensure it's a directory
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        return img_path, label

    def get_class_subset(self, class_name):
        """Returns a subset of the dataset containing only images of the given class."""
        if class_name not in self.class_to_idx:
            raise ValueError(f"Class '{class_name}' not found in dataset. Available classes: {CLASSES}")

        class_idx = self.class_to_idx[class_name]
        subset_indices = [i for i, label in enumerate(self.labels) if label == class_idx]

        # Create a subset of the dataset
        subset = [(self.image_paths[i], self.labels[i]) for i in subset_indices]
        return subset

class FewShotDataset(Dataset):
    def __init__(self, dataframe, image_dir):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.device = (
            torch.device("mps") if torch.backends.mps.is_available() else
            torch.device("cuda") if torch.cuda.is_available() else
            torch.device("cpu")
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["Image Path"]
        label = self.dataframe.iloc[idx]["Class"]
        label_tensor = torch.tensor([label], dtype=torch.long).to(self.device)
        
        return img_path, label_tensor
