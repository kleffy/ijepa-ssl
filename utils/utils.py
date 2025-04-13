"""
Utility functions for I-JEPA Self-Supervised Learning
"""
import os
import sys
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import torch
import logging

from dataset.animal_dataset import AnimalDataset

def setup_logging(output_dir='logs'):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
def setup_experiments_dir(config):
    experiment_name = f'{config.get('experiment_name', 'ijepa_experiment')}_{config.get('version', 'V1')}'
    config['output_dir'] = os.path.join(config.get('output_dir', 'experiments'), experiment_name)
    
    # Set device if it's not explicitly set in the config
    if 'device' not in config:
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ensure output directory exists
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # save config to output directory
    config_path = os.path.join(config['output_dir'], 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        

def update_ema(model, ema_model, decay):
    """Update the Exponential Moving Average (EMA) model"""
    with torch.no_grad():
        for param, ema_param in zip(model.parameters(), ema_model.parameters()):
            if param.device != ema_param.device:
                 ema_param.data = ema_param.data.to(param.device)
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def get_num_patches(image_size, patch_size):
    """Calculate the number of patches based on image and patch size"""
    return image_size // patch_size


def get_dataloader(config, val_ratio=0.1, shuffle=True):
    """Creates train and validation dataloaders using the AnimalDataset"""
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        full_dataset = AnimalDataset(root_dir=config['dataset_path'], transform=transform)
        if not full_dataset.classes or len(full_dataset) == 0:
            logging.error(f"No classes or samples found by AnimalDataset in {config['dataset_path']}.")
            return None, None, None, None

        # Split dataset into train and validation
        dataset_size = len(full_dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        logging.info(f"Dataset split: {train_size} training samples, {val_size} validation samples")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=shuffle,
            num_workers=4, 
            pin_memory=True, 
            drop_last=shuffle,
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config['batch_size'], 
            shuffle=False,  # No need to shuffle validation data
            num_workers=4, 
            pin_memory=True, 
            drop_last=False,
        )
        
        return train_loader, val_loader, full_dataset.classes, full_dataset
    
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return None, None, None, None

