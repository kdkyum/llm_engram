import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import re

def graft_top_k_diff(
    modelA: nn.Module, 
    modelB: nn.Module, 
    excluded_param_names: Optional[List[str]] = None,
    graft_percentage: float = 0.01
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Given two models (modelA as the pretrained model and modelB as the finetuned model) with the same architecture,
    compute absolute parameter differences, find the top k% largest differences, and replace those parameters in modelA
    with the corresponding parameters from modelB.
    
    Additionally, returns a report dictionary containing a summary of the grafting operation for each module (layer).
    
    Args:
        modelA: The pretrained model to be updated.
        modelB: The finetuned model whose parameters will be grafted.
        excluded_param_names: List of parameter name regex patterns to exclude from grafting.
        graft_percentage: Percentage of parameters to graft (default: 0.01 for top 1%).
    
    Returns:
        A tuple containing the updated modelA and a dictionary report. The report dictionary has layer names as keys
        and values as another dictionary with the keys "total", "grafted", and "percentage" indicating the total number
        of parameters in that layer, the count of parameters grafted, and the percentage grafted, respectively.
    """
    # Default to empty list if None
    excluded_param_names = excluded_param_names or []
    
    # List to hold absolute differences for threshold determination 
    diffs = []
    # List of parameter pairs: (paramA, diff, paramB, name)
    param_pairs = []
    
    for (nameA, paramA), (nameB, paramB) in zip(modelA.named_parameters(), modelB.named_parameters()):
        # Skip parameters that match any exclusion pattern
        should_exclude = False
        for pattern in excluded_param_names:
            if re.search(pattern, nameA):
                print(f"Excluding parameter from grafting: {nameA}")
                should_exclude = True
                break
        if should_exclude:
            continue

        if paramA.shape == paramB.shape:
            diff = (paramA.data - paramB.data).abs()
            diffs.append(diff.view(-1))
            param_pairs.append((paramA, diff, paramB, nameA))
        else:
            print(f"Warning: Shape mismatch for parameter {nameA}. Skipping grafting for this parameter.")
    
    if not diffs:
        print("No parameters eligible for grafting after applying exclusions!")
        return modelA, {}
    
    # Concatenate all differences and compute the threshold for the top graft_percentage
    all_diffs = torch.cat(diffs)
    threshold = torch.quantile(all_diffs, 1.0 - graft_percentage).item()
    
    print(f"Overall grafting threshold: {threshold:.6f}")
    
    total_all_params = all_diffs.numel()
    total_grafted_params = (all_diffs > threshold).sum().item()
    print(f"Total grafting: {total_grafted_params} out of {total_all_params} parameters "
          f"({total_grafted_params/total_all_params:.2%})")
    
    # Report dictionary with grafting details for each parameter (module)
    report: Dict[str, Any] = {}
    
    # Transplant parameters and update report dictionary
    for paramA, diff, paramB, name in param_pairs:
        layer_total = diff.numel()
        mask = diff > threshold
        graft_count = mask.sum().item()
        if graft_count > 0:
            grafted_percentage = graft_count / layer_total
            print(f"Layer {name}: Will graft {graft_count} out of {layer_total} parameters "
                  f"({grafted_percentage:.2%})")
            # Update the report dictionary
            report[name] = {
                "total": layer_total,
                "grafted": graft_count,
                "percentage": grafted_percentage,
            }
            # Transplant parameters from modelB to modelA
            paramA.data[mask] = paramB.data[mask]
        else:
            print(f"Layer {name}: No parameters grafted out of {layer_total}.")
            report[name] = {
                "total": layer_total,
                "grafted": 0,
                "percentage": 0.0,
            }
    
    return modelA, report

# Example usage:
if __name__ == "__main__":
    # Example exclusion patterns (regex)
    excluded_layers = [
        "embedding",    # Exclude embedding layers
        "\.bias$",      # Exclude all bias parameters
        "layer\.0\.",   # Exclude the first layer
        "classifier",   # Exclude classifier layers
        "lm_head",      # Exclude language model head
    ]
    
    # Assume modelA is the pretrained model and modelB is the finetuned model with identical architectures.
    # For demonstration, you would typically initialize or load your models.
    # modelA = ...  # Pretrained model
    # modelB = ...  # Finetuned model
    
    # Perform grafting: transplant the top graft_percentage diff parameters from modelB onto modelA.
    # updated_model, graft_report = graft_top_k_diff(modelA, modelB, excluded_param_names=excluded_layers, graft_percentage=0.01)
    
    # Print out the grafting report
    # for layer, details in graft_report.items():
    #     print(f"Layer {layer}: {details}")
    pass