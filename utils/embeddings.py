import logging
import numpy as np
import torch
from tqdm import tqdm

def generate_embeddings(encoder, dataloader, config):
    """Generates embeddings for the entire dataset using the trained encoder"""
    device = torch.device(config['device'])
    logging.info("Generating embeddings for evaluation...")
    encoder.eval()
    encoder.to(device)
    features = []
    labels = []

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Generating Embeddings"):
            if batch_data is None: continue
            images, lbls = batch_data
            images = images.to(device)

            if hasattr(encoder, 'forward_features'):
                 feats_all_tokens = encoder.forward_features(images)
                 num_prefix = getattr(encoder, 'num_prefix_tokens', 1)
                 if getattr(encoder, 'global_pool', None) == 'token' or num_prefix > 0:
                      if feats_all_tokens.shape[1] > 0:
                          feats = feats_all_tokens[:, 0]
                      else:
                          feats = torch.zeros((images.shape[0], encoder.embed_dim), device=device)
                 else:
                      patch_feats = feats_all_tokens[:, num_prefix:]
                      if patch_feats.shape[1] > 0:
                          feats = patch_feats.mean(dim=1)
                      else:
                          feats = torch.zeros((images.shape[0], encoder.embed_dim), device=device)
            else:
                 try:
                     patch_embeds = encoder.patch_embed(images)
                     feats = patch_embeds.mean(dim=1)
                 except Exception:
                     feats = torch.zeros((images.shape[0], encoder.embed_dim), device=device)

            features.append(feats.cpu().numpy())
            labels.append(lbls.cpu().numpy())

    if not features:
        return None, None

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    logging.info(f"Generated {features.shape[0]} embeddings with dimension {features.shape[1]}.")
    return features, labels