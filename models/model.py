import math
import random
import copy
import logging
import timm
import torch
import torch.nn as nn


def get_grid_size(image_size, patch_size):
    """Calculate the grid dimensions (height, width)"""
    grid_size = image_size // patch_size
    return (grid_size, grid_size)

def get_embed_dim(model):
    if hasattr(model, 'embed_dim'):
        return model.embed_dim
    elif hasattr(model, 'fc') and hasattr(model.fc, 'in_features'):
        return model.fc.in_features
    else:
        raise AttributeError("Model has neither embed_dim nor fc.in_features attribute.")


class IJEPA(nn.Module):
    """I-JEPA Model Implementation"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_size = config['patch_size']
        self.grid_size_h, self.grid_size_w = get_grid_size(config['image_size'], config['patch_size'])
        self.num_patches = self.grid_size_h * self.grid_size_w

        # Initialize encoders
        self.context_encoder = timm.create_model(config['model_name'], pretrained=config['pretrained'], num_classes=0)
        actual_embed_dim = get_embed_dim(self.context_encoder)
        embed_dim = config['embed_dim']
        if actual_embed_dim != embed_dim:
             logging.warning(f"Model {config['model_name']} embed_dim {actual_embed_dim} != config {embed_dim}. Using {actual_embed_dim}.")
             embed_dim = actual_embed_dim

        # Create target encoder as a copy of context encoder
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Check embedding dimensions
        predictor_embed_dim = config['predictor_embed_dim']
        if predictor_embed_dim != embed_dim:
             logging.warning(f"PREDICTOR_EMBED_DIM ({predictor_embed_dim}) differs from encoder EMBED_DIM ({embed_dim}). Ensure projections are intended.")

        # Determine number of attention heads
        try:
            encoder_num_heads = self.context_encoder.blocks[0].attn.num_heads
        except AttributeError:
            logging.warning("Could not determine num_heads from context_encoder.blocks[0].attn.num_heads.")
            head_dim = 64; encoder_num_heads = embed_dim // head_dim
            logging.warning(f"Falling back to calculated num_heads: {encoder_num_heads} (assuming head_dim={head_dim})")
            if embed_dim % head_dim != 0:
                logging.warning("Fallback num_heads calculation might be inaccurate.")

        # Initialize predictor
        self.predictor = nn.ModuleList([
            timm.models.vision_transformer.Block(
                dim=predictor_embed_dim,
                num_heads=encoder_num_heads,
                mlp_ratio=4.,
                qkv_bias=True
            ) for _ in range(config['predictor_depth'])
        ])
        self.predictor_pos_embed = nn.Parameter(torch.zeros(1, config['num_target_masks'] + 1, predictor_embed_dim))
        nn.init.trunc_normal_(self.predictor_pos_embed, std=.02)

        # Projection layers
        self.context_proj = nn.Linear(embed_dim, predictor_embed_dim) if embed_dim != predictor_embed_dim else nn.Identity()
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim) if predictor_embed_dim != embed_dim else nn.Identity()

    def forward_encoder(self, imgs, masks, encoder):
        """Forward pass through encoder with masks"""
        B = imgs.shape[0]
        x = encoder.forward_features(imgs)
        num_prefix_tokens = getattr(encoder, 'num_prefix_tokens', 1)
        patch_embeds = x[:, num_prefix_tokens:]
        _, N_patches_embed, D_embed = patch_embeds.shape

        if masks.shape != (B, N_patches_embed):
            raise ValueError(f"Mask shape mismatch: Got {masks.shape}, expected {(B, N_patches_embed)}")

        visible_embeds_list = []
        for i in range(B):
            visible_indices = masks[i].nonzero(as_tuple=False).flatten()
            if len(visible_indices) > 0:
                visible_embeds_list.append(patch_embeds[i, visible_indices])
            else:
                visible_embeds_list.append(torch.zeros((0, D_embed), device=imgs.device))
        return visible_embeds_list

    def forward_predictor(self, context_embeds_list, target_masks):
        """Forward pass through predictor"""
        B = len(context_embeds_list)
        device = target_masks.device
        context_summaries = []

        # Process context embeddings
        for embeds in context_embeds_list:
            if embeds.shape[0] > 0:
                summary = embeds.mean(dim=0, keepdim=True)
            else:
                summary = torch.zeros((1, self.context_encoder.embed_dim), device=device)
            context_summaries.append(self.context_proj(summary))

        batched_context_summary = torch.cat(context_summaries, dim=0)
        if batched_context_summary.ndim == 2:
            batched_context_summary = batched_context_summary.unsqueeze(1)

        # Add target placeholders and positional embeddings
        target_placeholders = self.predictor_pos_embed[:, 1:, :].expand(B, -1, -1)
        predictor_input = torch.cat([batched_context_summary, target_placeholders], dim=1)
        predictor_input = predictor_input + self.predictor_pos_embed

        # Forward through predictor blocks
        pred_embeds = predictor_input
        for blk in self.predictor:
            pred_embeds = blk(pred_embeds)

        # Extract and project target embeddings
        predicted_target_embeds = pred_embeds[:, 1:, :]
        predicted_target_embeds = self.predictor_proj(predicted_target_embeds)
        return predicted_target_embeds

    def forward(self, imgs, context_masks, target_masks):
        """Full model forward pass"""
        
        N_patches = self.num_patches
        N_targets = self.config['num_target_masks']
        device = imgs.device

        # Get target embeddings with no gradient
        with torch.no_grad():
            if hasattr(self.target_encoder, 'forward_features'):
                 target_embeds_all_tokens = self.target_encoder.forward_features(imgs)
                 num_prefix = getattr(self.target_encoder, 'num_prefix_tokens', 1)
                 if target_embeds_all_tokens.shape[1] <= num_prefix:
                     return torch.tensor(0.0, device=device, requires_grad=True)
                 target_embeds_all = target_embeds_all_tokens[:, num_prefix:]
            else:
                return torch.tensor(0.0, device=device, requires_grad=True)
            target_embeds_all = target_embeds_all.detach()

        # Forward through context encoder and predictor
        context_embeds_list = self.forward_encoder(imgs, context_masks, self.context_encoder)
        predicted_target_embeds = self.forward_predictor(context_embeds_list, target_masks)

        # Calculate loss
        loss = 0.0
        total_targets_predicted = 0
        if target_embeds_all.shape[1] != N_patches:
            logging.warning(f"N_patches mismatch in loss. Embeds: {target_embeds_all.shape[1]}, Expected: {N_patches}.")

        for i in range(N_targets):
            current_target_mask = target_masks[:, i, :]
            if current_target_mask.shape[1] != target_embeds_all.shape[1]:
                continue

            current_predicted = predicted_target_embeds[:, i, :]
            num_patches_in_mask = current_target_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mask_expanded = current_target_mask.unsqueeze(-1)
            true_target_sum = (target_embeds_all * mask_expanded).sum(dim=1)
            true_target_avg = true_target_sum / num_patches_in_mask

            mse_loss = nn.functional.mse_loss(current_predicted, true_target_avg.detach())
            loss += mse_loss
            total_targets_predicted += 1

        if total_targets_predicted > 0:
            loss /= total_targets_predicted
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        return loss


class MultiBlockMaskCollator:
    """Generates context and target masks for I-JEPA"""
    def __init__(self, config):
        self.config = config
        self.grid_size_h, self.grid_size_w = get_grid_size(config['image_size'], config['patch_size'])
        self.num_patches = self.grid_size_h * self.grid_size_w
        self.device = torch.device(config['device'])

    def __call__(self, batch):
        if batch is None: 
            return None, None, None
        
        images, _ = batch
        batch_size = images.shape[0]
        
        if batch_size == 0: 
            return None, None, None
        
        context_masks, target_masks = [], []
        
        for i in range(batch_size):
            context_scale = self.config['mask_scales'][0]
            context_mask_indices = self._generate_mask_indices(context_scale, i)
            context_mask = torch.zeros(self.num_patches, dtype=torch.bool)
            valid_indices = context_mask_indices[context_mask_indices < self.num_patches]
            context_mask[valid_indices] = True
            context_masks.append(context_mask)
            current_target_masks = []
            
            for _ in range(self.config['num_target_masks']):
                target_scale = self.config['mask_scales'][1]
                target_mask_indices = self._generate_mask_indices(target_scale, i)
                target_mask = torch.zeros(self.num_patches, dtype=torch.bool)
                valid_indices = target_mask_indices[target_mask_indices < self.num_patches]
                target_mask[valid_indices] = True
                current_target_masks.append(target_mask)
                
            target_masks.append(torch.stack(current_target_masks))
            
        batched_context_masks = torch.stack(context_masks).to(self.device, non_blocking=True)
        batched_target_masks = torch.stack(target_masks).to(self.device, non_blocking=True)
        
        return images.to(self.device, non_blocking=True), batched_context_masks, batched_target_masks

    def _generate_mask_indices(self, scale_params, item_index):
        """Generates indices for a single mask block"""
        mask_ratio = scale_params['ratio']
        min_aspect, max_aspect = scale_params['aspect_ratio_range']
        log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        for attempt in range(self.config['mask_generation_retries']):
            current_ratio = mask_ratio * random.uniform(0.9, 1.1)
            num_visible_patches = int(self.num_patches * current_ratio)
            num_visible_patches = max(1, min(num_visible_patches, self.num_patches - 1))

            aspect_ratio = math.exp(random.uniform(*log_aspect_ratio))
            h = max(1, int(round(math.sqrt(num_visible_patches / aspect_ratio))))
            w = max(1, int(round(math.sqrt(num_visible_patches * aspect_ratio))))

            if h <= self.grid_size_h and w <= self.grid_size_w:
                top = random.randint(0, self.grid_size_h - h)
                left = random.randint(0, self.grid_size_w - w)
                indices = torch.arange(self.num_patches).reshape(self.grid_size_h, self.grid_size_w)
                mask_indices = indices[top:top+h, left:left+w].flatten()
                return mask_indices

        logging.warning(f"Mask block generation failed after {self.config['mask_generation_retries']} retries (item {item_index}). Using random patches.")
        return torch.randperm(self.num_patches)[:num_visible_patches]