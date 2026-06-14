import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np

class TomatoLeafDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_image_paths_and_labels(dataset_dir):
    """
    Scans the dataset directory recursively for images in LI, MI, HI, FI.
    """
    class_mapping = {
        "LI": 0,  # Low Irrigated
        "MI": 1,  # Medium Irrigated
        "HI": 2,  # Highly Irrigated
        "FI": 3   # Fully Irrigated
    }
    
    file_paths = []
    labels = []
    
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    
    for class_name, label in class_mapping.items():
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_path):
            print(f"[WARNING] Path {class_path} does not exist.")
            continue
            
        class_files = []
        for ext in image_extensions:
            # Search recursively
            class_files.extend(glob.glob(os.path.join(class_path, "**", ext), recursive=True))
            
        print(f"Found {len(class_files)} files for class {class_name}")
        file_paths.extend(class_files)
        labels.extend([label] * len(class_files))
        
    return file_paths, labels

def get_transforms():
    """
    Returns data transforms for train, validation, and test datasets.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),  # Scales pixels to [0, 1] range
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Scales pixels to [0, 1] range
    ])
    
    return train_transform, val_test_transform

def plot_dataset_distribution(labels_train, labels_val, labels_test, base_dir):
    """
    Plots a stacked bar chart of the dataset split and saves it.
    """
    class_names = ["LI", "MI", "HI", "FI"]
    train_counts = [labels_train.count(i) for i in range(4)]
    val_counts = [labels_val.count(i) for i in range(4)]
    test_counts = [labels_test.count(i) for i in range(4)]
    
    x = np.arange(len(class_names))
    width = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bars
    p1 = ax.bar(x, train_counts, width, label='Training', color='skyblue')
    p2 = ax.bar(x, val_counts, width, bottom=train_counts, label='Validation', color='gold')
    p3 = ax.bar(x, test_counts, width, bottom=np.array(train_counts)+np.array(val_counts), label='Testing', color='salmon')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Split Distribution Across Classes')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    
    # Add data labels
    for c1, c2, c3, _x in zip(train_counts, val_counts, test_counts, x):
        ax.text(_x, c1/2, str(c1), ha='center', va='center', color='black', fontsize=9)
        ax.text(_x, c1 + c2/2, str(c2), ha='center', va='center', color='black', fontsize=9)
        ax.text(_x, c1 + c2 + c3/2, str(c3), ha='center', va='center', color='black', fontsize=9)
        # Total
        ax.text(_x, c1 + c2 + c3 + 100, str(c1+c2+c3), ha='center', va='bottom', color='black', fontweight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "dataset_distribution.png"))
    plt.close()
    print("Saved dataset split visualization to dataset_distribution.png")

def get_data_loaders(dataset_dir, batch_size=64, random_state=42):
    """
    Performs standard 70/30 train/test split.
    Then splits the 70% train into 90% training and 10% validation.
    Returns train, val, and test DataLoader instances.
    """
    file_paths, labels = get_image_paths_and_labels(dataset_dir)
    
    if len(file_paths) == 0:
        raise ValueError("No images found in the dataset directory.")
        
    # Split into 70% train+val and 30% test
    paths_train_val, paths_test, labels_train_val, labels_test = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=random_state
    )
    
    # Split 70% train+val into 90% train and 10% validation
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_train_val, labels_train_val, test_size=0.10, stratify=labels_train_val, random_state=random_state
    )
    
    print(f"Dataset split stats:")
    print(f"  Training:   {len(paths_train)} samples")
    print(f"  Validation: {len(paths_val)} samples")
    print(f"  Testing:    {len(paths_test)} samples")
    
    # Plot and save distribution
    base_dir = os.path.dirname(dataset_dir)
    plot_dataset_distribution(labels_train, labels_val, labels_test, base_dir)
    
    train_transform, val_test_transform = get_transforms()
    
    train_dataset = TomatoLeafDataset(paths_train, labels_train, transform=train_transform)
    val_dataset = TomatoLeafDataset(paths_val, labels_val, transform=val_test_transform)
    test_dataset = TomatoLeafDataset(paths_test, labels_test, transform=val_test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader, test_loader
