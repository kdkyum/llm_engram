#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple LLM models")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum number of samples to use from dataset")
    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples to use for evaluation")
    parser.add_argument("--bio_field", type=str, default="bioS", help="Which bio field to use for training")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--run_lr_sweep", action="store_true", help="Run learning rate sweep for each model")
    parser.add_argument("--base_output_dir", type=str, default="./model-output", help="Base directory to save models")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Define models to train
    models = [
        # GPT-2 models
        {"name": "gpt2", "type": "gpt2", "default_lr": 5e-5, "gradient_accumulation_steps": 1, "per_device_train_batch_size": 4},
        {"name": "gpt2-medium", "type": "gpt2", "default_lr": 5e-5, "gradient_accumulation_steps": 1, "per_device_train_batch_size": 4},
        {"name": "gpt2-large", "type": "gpt2", "default_lr": 5e-5, "gradient_accumulation_steps": 1, "per_device_train_batch_size": 4},
        {"name": "gpt2-xl", "type": "gpt2", "default_lr": 5e-5, "gradient_accumulation_steps": 1, "per_device_train_batch_size": 4},
        # GPT-J and GPT-NeoX models
        {"name": "EleutherAI/gpt-j-6B", "type": "gpt-j", "default_lr": 5e-5, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1},
        # {"name": "EleutherAI/gpt-neox-20b", "type": "gpt-neox", "default_lr": 1e-5, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1},
        
        # Large models with LORA only
        # {"name": "meta-llama/Llama-3.2-1B", "type": "llama", "default_lr": 5e-3, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1, "fp16": True},
        # {"name": "meta-llama/Llama-3.2-3B", "type": "llama", "default_lr": 5e-3, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1, "fp16": True},
        {"name": "meta-llama/Llama-3.1-8B", "type": "llama", "default_lr": 5e-3, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1, "fp16": True},
    ]
    
    # Define learning rates for hyperparameter sweep if enabled
    lr_values = []
    if args.run_lr_sweep:
        # For each model's default LR, create a range of values around it
        for model_info in models:
            default_lr = model_info["default_lr"]
            # Create a range: 1x, 2x, and 5x the default learning rate
            lr_values.extend([
                default_lr,
                default_lr * 2.0,
                default_lr * 5
            ])
        # Remove duplicates and sort
        lr_values = sorted(list(set(lr_values)))
    else:
        # Just use the default learning rate for each model
        for model_info in models:
            lr_values.append(model_info["default_lr"])
    
    # Results tracking
    results = {}
    
    # Train each model
    for model_info in models:
        model_name = model_info["name"]
        model_type = model_info["type"]
        gradient_accumulation_steps = model_info["gradient_accumulation_steps"]
        per_device_train_batch_size = model_info["per_device_train_batch_size"]
        fp16 = model_info.get("fp16", False)
        
        # Get short name for directories
        model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
        
        print(f"\n{'='*80}")
        print(f"Training model: {model_name}")
        print(f"{'='*80}")
        
        model_results = []
        
        # Run with each learning rate in our sweep
        for learning_rate in (lr_values if args.run_lr_sweep else [model_info["default_lr"]]):
            # Create unique output directory for this model + learning rate combination
            lr_str = f"{learning_rate:.1e}".replace("-0", "-").replace("+0", "+")
            output_dir = os.path.join(
                args.base_output_dir,
                f"{model_short_name}-{args.bio_field}-lr{lr_str}"
            )
            
            print(f"\n{'-'*60}")
            print(f"Training {model_name} with learning rate: {learning_rate}")
            print(f"Output directory: {output_dir}")
            print(f"{'-'*60}")
            
            # Build command
            cmd = [
                "python", "train_model.py",  # Use our new script that handles bfloat16 correctly
                "--model_name_or_path", model_name,
                "--model_type", model_type,
                "--learning_rate", str(learning_rate),
                "--num_train_epochs", str(args.num_train_epochs),
                "--per_device_train_batch_size", str(per_device_train_batch_size),
                "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                "--max_samples", str(args.max_samples),
                "--eval_samples", str(args.eval_samples),
                "--bio_field", args.bio_field,
                "--output_dir", output_dir,
                "--wandb_run_name", f"{model_short_name}-{args.bio_field}-lr{lr_str}",
                "--freeze_embeddings"  # Always freeze embeddings to save memory
            ]
            
            if args.fp16 or fp16:
                cmd.append("--bf16")  # Use bf16 instead of fp16
                
            if args.lora:
                cmd.append("--lora")
            
            # Run training
            try:
                start_time = time.time()
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                end_time = time.time()
                
                # Extract accuracy from the output
                output = result.stdout
                final_accuracy = None
                if "Final Overall Accuracy" in output:
                    try:
                        final_accuracy_line = [line for line in output.split('\n') if "Final Overall Accuracy" in line][0]
                        final_accuracy = float(final_accuracy_line.split(':')[-1].strip())
                    except (IndexError, ValueError):
                        pass
                
                # Extract best accuracy from the output
                best_accuracy = None
                if "Best Overall Accuracy" in output:
                    try:
                        best_accuracy_line = [line for line in output.split('\n') if "Best Overall Accuracy" in line][0]
                        best_accuracy = float(best_accuracy_line.split(':')[-1].strip())
                    except (IndexError, ValueError):
                        pass
                
                # Calculate duration
                duration_seconds = end_time - start_time
                hours, remainder = divmod(duration_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                duration_str = f"{int(hours)}:{int(minutes):02d}:{int(seconds):02d}"
                
                # Record results
                run_result = {
                    "learning_rate": learning_rate,
                    "final_accuracy": final_accuracy,
                    "best_accuracy": best_accuracy,
                    "duration": duration_str,
                    "output_dir": output_dir
                }
                model_results.append(run_result)
                
                print(f"Successfully trained {model_name} with lr={learning_rate}")
                print(f"Final Accuracy: {final_accuracy}")
                print(f"Best Accuracy: {best_accuracy}")
                print(f"Training duration: {duration_str}")
                
            except subprocess.CalledProcessError as e:
                print(f"Error training {model_name} with lr={learning_rate}: {e}")
                print(f"STDOUT: {e.stdout}")
                print(f"STDERR: {e.stderr}")
                
                # Record failure
                model_results.append({
                    "learning_rate": learning_rate,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Store all results for this model
        results[model_name] = model_results
    
    # Print summary of results
    print("\n\n")
    print("=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("-" * 80)
        
        # Sort results by best accuracy (descending)
        sorted_results = sorted(
            [r for r in model_results if "best_accuracy" in r and r["best_accuracy"] is not None],
            key=lambda x: x["best_accuracy"], 
            reverse=True
        )
        
        if sorted_results:
            print(f"{'Learning Rate':<15} | {'Best Accuracy':<15} | {'Final Accuracy':<15} | {'Duration':<15} | {'Output Dir'}")
            print("-" * 80)
            
            for result in sorted_results:
                print(f"{result['learning_rate']:<15.2e} | "
                      f"{result['best_accuracy']:<15.4f} | "
                      f"{result['final_accuracy']:<15.4f} | "
                      f"{result['duration']:<15} | "
                      f"{result['output_dir']}")
        else:
            print("No successful runs for this model")
            
            # Print failed runs
            failed_runs = [r for r in model_results if "status" in r and r["status"] == "failed"]
            if failed_runs:
                print("\nFailed runs:")
                for failed in failed_runs:
                    print(f"  - LR {failed['learning_rate']}: {failed['error']}")
    
    # Save results to file
    results_file = os.path.join(args.base_output_dir, "training_results.txt")
    with open(results_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 80 + "\n")
            
            sorted_results = sorted(
                [r for r in model_results if "best_accuracy" in r and r["best_accuracy"] is not None],
                key=lambda x: x["best_accuracy"], 
                reverse=True
            )
            
            if sorted_results:
                f.write(f"{'Learning Rate':<15} | {'Best Accuracy':<15} | {'Final Accuracy':<15} | {'Duration':<15} | {'Output Dir'}\n")
                f.write("-" * 80 + "\n")
                
                for result in sorted_results:
                    f.write(f"{result['learning_rate']:<15.2e} | "
                          f"{result['best_accuracy']:<15.4f} | "
                          f"{result['final_accuracy']:<15.4f} | "
                          f"{result['duration']:<15} | "
                          f"{result['output_dir']}\n")
            else:
                f.write("No successful runs for this model\n")
                
                # Log failed runs
                failed_runs = [r for r in model_results if "status" in r and r["status"] == "failed"]
                if failed_runs:
                    f.write("\nFailed runs:\n")
                    for failed in failed_runs:
                        f.write(f"  - LR {failed['learning_rate']}: {failed['error']}\n")
            
            f.write("\n\n")
    
    print(f"\nDetailed training results saved to: {results_file}")
    print("\nTraining of all models complete!")

if __name__ == "__main__":
    main()