#!/usr/bin/env python3
import os
import torch

def setup_offline_mode(args=None):
    """Configure environment for offline operation on compute nodes"""
    
    # Define default cache paths if not specified
    default_cache_dir = "/ptmp/kdkyum/huggingface_cache"
    
    # Set cache directories if not already set
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = f"{default_cache_dir}/hf_home"
    if "HF_DATASETS_CACHE" not in os.environ:
        os.environ["HF_DATASETS_CACHE"] = f"{default_cache_dir}/datasets" 
    
    # Configure Weights & Biases for offline use if not already set
    if "WANDB_MODE" not in os.environ:
        os.environ["WANDB_MODE"] = "offline"
    
    # Set optimal PyTorch settings for A100 GPUs
    torch.backends.cudnn.benchmark = True
    
    # Print configuration
    print("\n===== 🔄 OFFLINE MODE CONFIGURATION =====")
    print(f"🔄 HF_HOME: {os.environ['HF_HOME']}")
    print(f"🔄 HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"🔄 WANDB_MODE: {os.environ.get('WANDB_MODE', 'not set')}")
    print(f"🔄 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔄 GPU Count: {torch.cuda.device_count()}")
        print(f"🔄 GPU Model: {torch.cuda.get_device_name(0)}")
    print("=========================================\n")
    
    return default_cache_dir