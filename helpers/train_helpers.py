#!/usr/bin/env python3
import torch

def get_device_fix_patch():
    """
    Monkey patch the fixed_cross_entropy function in transformers to handle multi-device scenarios
    """
    from transformers.loss import loss_utils
    
    original_fixed_cross_entropy = loss_utils.fixed_cross_entropy
    
    def patched_fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs):
        """
        Patched version that ensures tensors are on the same device before operations
        """
        import torch.nn.functional as F
        
        # Get target device from logits
        target_device = logits.device
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index, reduction=reduction)
        
        # Move num_items_in_batch to the device of loss if it's a tensor
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(target_device)

        if reduction == "sum":
            loss = loss / num_items_in_batch
        return loss
    
    # Patch the function
    loss_utils.fixed_cross_entropy = patched_fixed_cross_entropy