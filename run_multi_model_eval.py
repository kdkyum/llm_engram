#!/usr/bin/env python3
import os
import subprocess
import argparse
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation on multiple LLM models")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Define models to evaluate
    models = [
        # GPT-2 models
        {"name": "gpt2", "type": "gpt2", "path": "gpt2"},
        {"name": "gpt2-medium", "type": "gpt2", "path": "gpt2-medium"},
        {"name": "gpt2-large", "type": "gpt2", "path": "gpt2-large"},
        {"name": "gpt2-xl", "type": "gpt2", "path": "gpt2-xl"},
        
        # Fine-tuned GPT-2 XL model - replace with your actual path
        {"name": "gpt2-xl-finetuned", "type": "gpt2", "path": "./gpt2-xl-bios"},
        
        # GPT-J
        {"name": "gpt-j-6b", "type": "gpt-j", "path": "EleutherAI/gpt-j-6b"},
        
        # GPT-NeoX
        {"name": "gpt-neox-20b", "type": "gpt-neox", "path": "EleutherAI/gpt-neox-20b"},
        
        # Llama models
        {"name": "llama-3.2-1b", "type": "llama", "path": "meta-llama/Llama-3.2-1B"},
        {"name": "llama-3.2-3b", "type": "llama", "path": "meta-llama/Llama-3.2-3B"},
        {"name": "llama-3.1-8b", "type": "llama", "path": "meta-llama/Llama-3.1-8B"}
    ]
    
    # Prepare results container
    all_results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "fp16": args.fp16,
            "seed": args.seed
        },
        "results": {}
    }
    
    # Run evaluation for each model
    for model_info in models:
        model_name = model_info["name"]
        model_type = model_info["type"]
        model_path = model_info["path"]
        
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model_name}")
        print(f"{'='*50}")
        
        # Build command
        cmd = [
            "python", "evaluate_qa.py",
            "--model_path", model_path,
            "--model_type", model_type,
            "--num_samples", str(args.num_samples),
            "--batch_size", str(args.batch_size),
            "--seed", str(args.seed)
        ]
        
        if args.fp16:
            cmd.append("--fp16")
        
        # Run evaluation
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Extract results
            results_section = output.split("===== RESULTS =====")[1].strip()
            results_lines = results_section.split("\n")
            
            model_results = {}
            for line in results_lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = float(value.strip())
                    model_results[key] = value
            
            # Store results
            all_results["results"][model_name] = model_results
            
            # Save incremental results to file
            results_file = os.path.join(args.results_dir, "model_evaluation_results.json")
            with open(results_file, "w") as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Successfully evaluated {model_name}")
            print(f"Overall accuracy: {model_results.get('overall', 'N/A')}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model_name}: {e}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
    
    # Print summary of results
    print("\n\n")
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Model Name':<20} | {'Overall Accuracy':<20}")
    print("-" * 80)
    
    for model_name, results in all_results["results"].items():
        print(f"{model_name:<20} | {results.get('overall', 'N/A'):<20.4f}")
    
    print("\nComplete results saved to:", os.path.join(args.results_dir, "model_evaluation_results.json"))

if __name__ == "__main__":
    main()