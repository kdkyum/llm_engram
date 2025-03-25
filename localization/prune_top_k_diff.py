import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import re

def prune_top_k_diff(
    modelA: nn.Module, 
    modelB: nn.Module, 
    excluded_param_names: Optional[List[str]] = None,
    prune_percentage: float = 0.01
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Given two models (modelA and modelB) with the same architecture,
    compute the absolute parameter differences, find the top k% largest differences,
    and set those parameters in modelA to zero.
    
    Additionally, returns a report dictionary containing a summary of the pruning operation for each module (layer).
    
    Args:
        modelA: The model to be pruned.
        modelB: The reference model.
        excluded_param_names: List of regex patterns for parameter names to exclude from pruning.
        prune_percentage: Percentage of parameters to prune (default: 0.01 for top 1%).
    
    Returns:
        A tuple containing:
            - The modified modelA with pruned parameters.
            - A report dictionary where keys are parameter names and values are dicts containing 
              "total", "pruned", and "percentage" indicating the total number of parameters,
              the number pruned, and the fraction pruned for that parameter.
    """
    # Default to empty list if None
    excluded_param_names = excluded_param_names or []

    # Lists to collect parameter differences for threshold calculation and mapping for pruning
    diffs = []
    param_pairs = []  # Each entry: (paramA, diff, name)
    
    for (nameA, paramA), (nameB, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        # Skip parameters that match any exclusion patterns
        should_exclude = False
        for pattern in excluded_param_names:
            if re.search(pattern, nameA):
                print(f"Excluding parameter: {nameA}")
                should_exclude = True
                break
        if should_exclude:
            continue
        
        if paramA.shape == paramB.shape:
            diff = (paramA.data - paramB.data).abs()
            diffs.append(diff.view(-1))
            param_pairs.append((paramA, diff, nameA))
        else:
            print(f"Warning: Shape mismatch for parameter {nameA}. Skipping pruning for this parameter.")
    
    if not diffs:
        print("No parameters to prune after exclusions!")
        return modelA, {}
    
    # Concatenate all differences to compute the global threshold
    all_diffs = torch.cat(diffs)
    threshold = torch.quantile(all_diffs, 1.0 - prune_percentage).item()
    print(f"Overall pruning threshold: {threshold:.6f}")

    total_all_params = all_diffs.numel()
    total_pruned_params = (all_diffs > threshold).sum().item()
    print(f"Total pruning: {total_pruned_params} out of {total_all_params} parameters "
          f"({total_pruned_params/total_all_params:.2%})")
    
    # Initialize report dictionary to record stats for each parameter (module)
    report: Dict[str, Any] = {}
    
    # Prune selected parameters in modelA based on threshold and update report dictionary
    for paramA, diff, name in param_pairs:
        layer_total = diff.numel()
        mask = diff > threshold
        pruned_count = mask.sum().item()
        if pruned_count > 0:
            pruned_percentage = pruned_count / layer_total
            print(f"Layer {name}: Pruning {pruned_count} out of {layer_total} parameters "
                  f"({pruned_percentage:.2%})")
            paramA.data[mask] = 0.0
            report[name] = {"total": layer_total, "pruned": pruned_count, "percentage": pruned_percentage}
        else:
            print(f"Layer {name}: No parameters pruned out of {layer_total}.")
            report[name] = {"total": layer_total, "pruned": 0, "percentage": 0.0}
    
    return modelA, report

# Example usage:
if __name__ == "__main__":
    # Example of excluded parameter patterns (e.g., exclude embeddings, biases, classifier layers, etc.)
    excluded_layers = [
        "embedding",    # Exclude embedding layers
        "\.bias$",      # Exclude all bias parameters
        "layer\.0\.",   # Exclude the first layer
        "classifier",   # Exclude classifier layers
        "lm_head",      # Exclude language model head
    ]
    
    # Here you would normally load or instantiate your models with the same architecture:
    # modelA = ...  # Pretrained model
    # modelB = ...  # Reference model for pruning
    
    # For demo purposes, you can create simple dummy models with identical architectures.
    # updated_model, prune_report = prune_top_k_diff(modelA, modelB, excluded_param_names=excluded_layers, prune_percentage=0.01)
    # print(prune_report)
    pass