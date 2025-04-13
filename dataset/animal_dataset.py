# --- Dataset Class ---
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class AnimalDataset(Dataset):
    """Custom Dataset for loading animal images"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
         
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        
        if not self.classes:
            raise FileNotFoundError(f"No class subdirectories found in {root_dir}")
        
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            
            if not os.path.isdir(class_dir): 
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')) and not img_name.startswith('.'):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))
        
        if not self.samples:
            raise FileNotFoundError(f"No valid image files found in subdirectories of {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
            
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        if image is not None and not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
             
        if image is None:
            return None, None
        
        return image, label
    