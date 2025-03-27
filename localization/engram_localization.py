import os
import gc
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze weight changes to localize engrams in LLMs")
    parser.add_argument("--pretrained", type=str, default="meta-llama/Llama-2-13b-hf",
                        help="Path or name of the pretrained model")
    parser.add_argument("--finetuned", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--output_dir", type=str, default="output_plots",
                        help="Directory to save plots and results")
    parser.add_argument("--sample_size", type=int, default=50000,
                        help="Sample size for threshold estimation")
    parser.add_argument("--thresholds", type=str, default="0.1,0.05,0.01,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9",
                        help="Comma-separated list of thresholds to analyze (top k% changes)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for computation")
    return parser.parse_args()

def setup_output_directory(output_dir, model_name):
    """Create output directory structure based on model name"""
    model_short_name = model_name.split('/')[-2]
    output_path = os.path.join(output_dir, f"{model_short_name}")
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories for different types of plots
    os.makedirs(os.path.join(output_path, "layer_analysis"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "component_analysis"), exist_ok=True)
    
    return output_path

def load_models(pretrained_path, finetuned_path, device="cpu"):
    """Load pretrained and finetuned models"""
    print(f"Loading pretrained model from {pretrained_path}...")
    model_pretrain = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=False,
    )
    
    print(f"Loading fine-tuned model from {finetuned_path}...")
    model_finetune = AutoModelForCausalLM.from_pretrained(
        finetuned_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        use_cache=False,
    )
    
    model_pretrain.eval()
    model_finetune.eval()
    
    return model_pretrain, model_finetune

def estimate_thresholds(model_pretrain, model_finetune, sample_size=50000):
    """Estimate thresholds for detecting significant weight changes"""
    print(f"Estimating thresholds using {sample_size} samples...")
    gc.collect()
    torch.cuda.empty_cache()
    
    samples = []
    sampled_elements = 0
    
    # Get only relevant keys (from layers with mlp or attn)
    relevant_keys = [k for k in model_finetune.state_dict().keys() 
                    if k in model_pretrain.state_dict() and 
                    "model.layers." in k and ("mlp" in k or "attn" in k)]
    
    # Randomly shuffle keys for better sampling
    import random
    random.shuffle(relevant_keys)
    
    for key in relevant_keys:
        if sampled_elements >= sample_size:
            break
            
        # Process on CPU to save GPU memory
        param_finetune = model_finetune.state_dict()[key].cpu()
        param_pretrain = model_pretrain.state_dict()[key].cpu()
        
        diff = torch.abs(param_finetune - param_pretrain).flatten()
        
        # Take only what we need for the sample
        num_to_sample = min(diff.numel(), sample_size - sampled_elements)
        if num_to_sample < diff.numel():
            indices = torch.randperm(diff.numel())[:num_to_sample]
            samples.append(diff[indices])
        else:
            samples.append(diff)
        
        sampled_elements += num_to_sample
        
        # Free memory
        del param_finetune, param_pretrain, diff
        gc.collect()
    
    samples = torch.cat(samples).float()
    return samples, relevant_keys

def analyze_layer_changes(model_pretrain, model_finetune, relevant_keys, threshold, output_path):
    """Analyze changes per layer above the specified threshold"""
    print(f"Analyzing layer changes at threshold {threshold:.10f}...")
    
    # Count parameters above threshold per layer
    layer_changes = {}
    
    for key in relevant_keys:
        parts = key.split(".")
        layer_id = parts[2]
        
        if layer_id not in layer_changes:
            layer_changes[layer_id] = {"attn": 0, "mlp": 0}
        
        # Process on CPU
        param_finetune = model_finetune.state_dict()[key].cpu()
        param_pretrain = model_pretrain.state_dict()[key].cpu()
        
        # Count differences exceeding threshold
        count = (torch.abs(param_finetune - param_pretrain) >= threshold).sum().item()
        
        # Update counts
        if "mlp" in key:
            layer_changes[layer_id]["mlp"] += count
        elif "attn" in key:
            layer_changes[layer_id]["attn"] += count
        
        # Free memory
        del param_finetune, param_pretrain
        gc.collect()
    
    # Save numeric results
    result_file = os.path.join(output_path, "layer_analysis", f"threshold_{threshold:.10e}.csv")
    with open(result_file, 'w') as f:
        f.write("layer,attention_changes,mlp_changes\n")
        for layer in sorted(layer_changes.keys(), key=lambda x: int(x)):
            f.write(f"{layer},{layer_changes[layer]['attn']},{layer_changes[layer]['mlp']}\n")
    
    # Plot results
    layers = sorted(layer_changes.keys(), key=lambda x: int(x))
    attn_counts = [layer_changes[layer]["attn"] for layer in layers]
    mlp_counts = [layer_changes[layer]["mlp"] for layer in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, attn_counts, width, label='Attention')
    ax.bar(x + width/2, mlp_counts, width, label='MLP')
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Count of changed parameters')
    ax.set_title(f'Localization of Changed Parameters (threshold={threshold:.6f})')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    
    plot_file = os.path.join(output_path, "layer_analysis", f"threshold_{threshold:.10e}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return layer_changes

def analyze_attention_components(model_pretrain, model_finetune, relevant_keys, threshold, output_path, num_layers=40):
    """Analyze changes in attention components above threshold"""
    print(f"Analyzing attention components at threshold {threshold:.10f}...")
    
    # Initialize dictionaries to track changes by component
    layer_changes = {
        'q_proj': {layer: 0 for layer in range(num_layers)},
        'k_proj': {layer: 0 for layer in range(num_layers)},
        'v_proj': {layer: 0 for layer in range(num_layers)},
        'o_proj': {layer: 0 for layer in range(num_layers)}
    }
    
    # Process all attention keys
    for key in relevant_keys:
        if "self_attn" not in key:
            continue
            
        parts = key.split(".")
        layer_id = int(parts[2])
        if layer_id >= num_layers:  # Skip if layer is out of range
            continue
            
        component = parts[4]  # Will be q_proj, k_proj, v_proj, or o_proj
        
        # Skip if not one of our target components
        if component not in layer_changes:
            continue
        
        # Process on CPU
        param_finetune = model_finetune.state_dict()[key].cpu()
        param_pretrain = model_pretrain.state_dict()[key].cpu()
        
        # Count differences exceeding threshold
        count = (torch.abs(param_finetune - param_pretrain) >= threshold).sum().item()
        layer_changes[component][layer_id] += count
        
        # Free memory
        del param_finetune, param_pretrain
        gc.collect()
    
    # Save numeric results
    result_file = os.path.join(output_path, "component_analysis", f"attn_components_{threshold:.10e}.csv")
    with open(result_file, 'w') as f:
        f.write("layer,q_proj,k_proj,v_proj,o_proj\n")
        for layer in range(num_layers):
            f.write(f"{layer},{layer_changes['q_proj'][layer]},{layer_changes['k_proj'][layer]}," +
                   f"{layer_changes['v_proj'][layer]},{layer_changes['o_proj'][layer]}\n")
    
    # Create plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 12))
    axs = axs.flatten()
    
    layers = list(range(num_layers))
    x = np.arange(len(layers))
    
    # Plot each component
    titles = ['Query Projection', 'Key Projection', 'Value Projection', 'Output Projection']
    components = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    
    for i, (component, title) in enumerate(zip(components, titles)):
        counts = [layer_changes[component][layer] for layer in layers]
        axs[i].bar(x, counts, width=0.7)
        axs[i].set_xlabel('Layer')
        axs[i].set_ylabel('Count of parameters > threshold')
        axs[i].set_title(f'{title} Changes')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(layers)
    
    plt.tight_layout()
    plt.suptitle(f'Attention Component Changes Above Threshold {threshold:.6f}', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    plot_file = os.path.join(output_path, "component_analysis", f"attn_components_{threshold:.10e}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return layer_changes

def analyze_mlp_components(model_pretrain, model_finetune, relevant_keys, threshold, output_path, num_layers=40):
    """Analyze changes in MLP components above threshold"""
    print(f"Analyzing MLP components at threshold {threshold:.10f}...")
    
    # Initialize dictionaries to track changes by component
    mlp_changes = {
        'gate_proj': {layer: 0 for layer in range(num_layers)},
        'up_proj': {layer: 0 for layer in range(num_layers)},
        'down_proj': {layer: 0 for layer in range(num_layers)}
    }
    
    # Process all MLP keys
    for key in relevant_keys:
        if "mlp" not in key:
            continue
            
        parts = key.split(".")
        layer_id = int(parts[2])
        if layer_id >= num_layers:  # Skip if layer is out of range
            continue
            
        component = parts[4]  # Will be gate_proj, up_proj, or down_proj
        
        # Skip if not one of our target components
        if component not in mlp_changes:
            continue
        
        # Process on CPU
        param_finetune = model_finetune.state_dict()[key].cpu()
        param_pretrain = model_pretrain.state_dict()[key].cpu()
        
        # Count differences exceeding threshold
        count = (torch.abs(param_finetune - param_pretrain) >= threshold).sum().item()
        mlp_changes[component][layer_id] += count
        
        # Free memory
        del param_finetune, param_pretrain
        gc.collect()
    
    # Save numeric results
    result_file = os.path.join(output_path, "component_analysis", f"mlp_components_{threshold:.10e}.csv")
    with open(result_file, 'w') as f:
        f.write("layer,gate_proj,up_proj,down_proj\n")
        for layer in range(num_layers):
            f.write(f"{layer},{mlp_changes['gate_proj'][layer]},{mlp_changes['up_proj'][layer]}," +
                   f"{mlp_changes['down_proj'][layer]}\n")
    
    # Create plot
    fig, axs = plt.subplots(3, 1, figsize=(20, 15))
    axs = axs.flatten()
    
    layers = list(range(num_layers))
    x = np.arange(len(layers))
    
    # Plot each component
    titles = ['Gate Projection', 'Up Projection', 'Down Projection']
    components = ['gate_proj', 'up_proj', 'down_proj']
    
    for i, (component, title) in enumerate(zip(components, titles)):
        counts = [mlp_changes[component][layer] for layer in layers]
        axs[i].bar(x, counts, width=0.7)
        axs[i].set_xlabel('Layer')
        axs[i].set_ylabel('Count of parameters > threshold')
        axs[i].set_title(f'{title} Changes')
        axs[i].set_xticks(x)
        axs[i].set_xticklabels(layers)
    
    plt.tight_layout()
    plt.suptitle(f'MLP Component Changes Above Threshold {threshold:.6f}', fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    plot_file = os.path.join(output_path, "component_analysis", f"mlp_components_{threshold:.10e}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return mlp_changes

def save_metadata(args, output_path, start_time):
    """Save analysis metadata to a file"""
    end_time = datetime.now()
    duration = end_time - start_time
    
    with open(os.path.join(output_path, "analysis_metadata.txt"), 'w') as f:
        f.write(f"Engram Localization Analysis\n")
        f.write(f"===========================\n\n")
        f.write(f"Pre-trained model: {args.pretrained}\n")
        f.write(f"Fine-tuned model: {args.finetuned}\n")
        f.write(f"Analysis date: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Sample size: {args.sample_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Thresholds analyzed: {args.thresholds}\n")

def main():
    """Main function to execute the engram localization analysis"""
    args = parse_args()
    start_time = datetime.now()
    
    # Set up output directory
    output_path = setup_output_directory(args.output_dir, args.finetuned)
    print(f"Results will be saved to {output_path}")
    
    # Load models
    model_pretrain, model_finetune = load_models(args.pretrained, args.finetuned, args.device)
    
    # Estimate thresholds
    samples, relevant_keys = estimate_thresholds(model_pretrain, model_finetune, args.sample_size)
    
    # Parse thresholds
    thresholds = [float(t) for t in args.thresholds.split(',')]
    
    # For each threshold, analyze layer changes
    for k in thresholds:
        # For percentage-based thresholds
        if k < 1:
            threshold = torch.quantile(samples, 1 - k).item()
        else: 
            # For direct threshold values
            threshold = k
            
        print(f"Analyzing with threshold: {threshold}")
        
        # Analyze layer changes
        analyze_layer_changes(model_pretrain, model_finetune, relevant_keys, threshold, output_path)
        
        # Analyze component changes
        analyze_attention_components(model_pretrain, model_finetune, relevant_keys, threshold, output_path)
        analyze_mlp_components(model_pretrain, model_finetune, relevant_keys, threshold, output_path)
    
    # Save metadata about the analysis
    save_metadata(args, output_path, start_time)
    
    print(f"Analysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
