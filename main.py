import argparse
import logging
import os
import sys
import yaml
import torch
import torch.optim as optim
from tqdm import tqdm

from models.model import IJEPA, MultiBlockMaskCollator
from utils.eval import evaluate_ssl_model
from utils.utils import setup_experiments_dir, setup_logging, update_ema, get_dataloader
from utils.viz import visualize_features


def train_ijepa(config):
    """Main training function for I-JEPA with validation-based early stopping"""
    device = torch.device(config['device'])
    logging.info(f"Using device: {device}")
   
    # Get train and validation dataloaders
    val_ratio = config.get('validation_ratio', 0.1)
    train_loader, val_loader, class_names, _ = get_dataloader(
        config, val_ratio=val_ratio, shuffle=True
    )
    
    if train_loader is None:
        return None, None
   
    mask_collator = MultiBlockMaskCollator(config)
    model = IJEPA(config).to(device)
   
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
   
    logging.info("Starting I-JEPA training...")
    model_save_path = os.path.join(config['output_dir'], 'ijepa_model.pth')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Early stopping parameters
    patience = config.get('early_stopping_patience', 10)
    min_delta = config.get('early_stopping_min_delta', 0.0001)
    wait = 0
    best_val_loss = float('inf')
   
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        model.target_encoder.eval()
        total_train_loss = 0.0
        train_batches = 0
   
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch_data in progress_bar:
            processed_data = mask_collator(batch_data)
            if processed_data is None: continue
           
            images, context_masks, target_masks = processed_data
            optimizer.zero_grad()
            loss = model(images, context_masks, target_masks)
           
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss) and loss.requires_grad:
                loss.backward()
                optimizer.step()
                update_ema(model.context_encoder, model.target_encoder, config['ema_decay'])
                total_train_loss += loss.item()
                train_batches += 1
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else 0.0
        
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for batch_data in progress_bar:
                processed_data = mask_collator(batch_data)
                if processed_data is None: continue
                
                images, context_masks, target_masks = processed_data
                val_loss = model(images, context_masks, target_masks)
                
                if val_loss is not None and not torch.isnan(val_loss) and not torch.isinf(val_loss):
                    total_val_loss += val_loss.item()
                    val_batches += 1
                    progress_bar.set_postfix(loss=f"{val_loss.item():.4f}")
        
        avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Check if model improved on validation set
        if avg_val_loss < best_val_loss - min_delta:
            logging.info(f"Validation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            torch.save(model.context_encoder.state_dict(), model_save_path)
            wait = 0
        else:
            wait += 1
            logging.info(f"Validation loss did not improve from {best_val_loss:.4f}. Patience: {wait}/{patience}")
            if wait >= patience:
                logging.info(f"Early stopping triggered after {epoch+1} epochs")
                break
   
    logging.info(f"Training finished. Best model saved to {model_save_path}")
    
    # Load the best model
    best_encoder = model.context_encoder
    try:
        state_dict = torch.load(model_save_path, map_location=device)
        best_encoder.load_state_dict(state_dict)
    except Exception as e:
        logging.error(f"Error loading best model: {e}")
    
    return best_encoder, class_names


def main(config):
    """Main function to load config and run the pipeline"""
    setup_experiments_dir(config)
    
    setup_logging(output_dir=config.get('output_dir', 'logs'))
    
    # Complete pipeline: train, evaluate, visualize
    logging.info("Starting I-JEPA pipeline")
    
    # Train model
    encoder, class_names = train_ijepa(config)
    
    if encoder is None:
        logging.error("Training failed.")
        sys.exit(1)
    
    # Evaluate model
    _, _, _, dataset = get_dataloader(config, shuffle=False)
    
    if dataset is not None:
        evaluate_ssl_model(encoder, config, class_names, dataset)
    else:
        logging.error("Evaluation failed: could not load dataset.")
    
    # Visualize features
    visualize_features(encoder, config, class_names)
    
    logging.info("I-JEPA pipeline completed successfully")



if __name__ == "__main__":
    config_path = '/vol/research/RobotFarming/Projects/ijepa_ssl/config/config.yaml'
    parser = argparse.ArgumentParser(description='I-JEPA Self-Supervised Learning')
    parser.add_argument('--config', type=str, default=config_path, help='Path to YAML config file')
    args = parser.parse_args()
    
    # Load config from YAML file
    if not os.path.exists(args.config):
        logging.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    main(config)