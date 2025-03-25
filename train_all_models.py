#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple LLM models")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=10, help="Maximum number of samples to use from dataset")
    parser.add_argument("--eval_samples", type=int, default=10, help="Number of samples to use for evaluation")
    parser.add_argument("--bio_field", type=str, default="bioS", help="Which bio field to use for training")
    parser.add_argument("--run_lr_sweep", action="store_true", help="Run learning rate sweep for each model")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--base_output_dir", type=str, default="./model-output", help="Base directory to save models")
    parser.add_argument("--mcq_percentage", type=int, default=0, help="Percentage of MCQ evaluation examples to include in training data (0-100)")
    parser.add_argument("--mcq_with_bios", action="store_true", help="Include bioS text with MCQ examples (if not set, only MCQ text is used)")
    parser.add_argument("--shuffle_eval_choices", action="store_true", help="Shuffle MCQ choices during evaluation to test for overfitting")
    return parser.parse_args()

def main():
    args = parse_args()

    # Define all possible bio fields
    bio_fields = ["bioS_multi5_permutes"] #]
    
    # Define models to train
    models = [
        # GPT-2 models
        # {"name": "gpt2", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 10},
        # {"name": "gpt2-medium", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 10},
        # {"name": "gpt2-large", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 10},
        # {"name": "gpt2-xl", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 10},
        # {"name": "meta-llama/Llama-3.2-1B", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 1},
        # {"name": "meta-llama/Llama-3.2-3B", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        # {"name": "EleutherAI/gpt-j-6B", "type": "gpt-j", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-7b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-3.1-8B", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-7B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-13b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-13B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        #meta-llama/Llama-2-70b-hf
        # {"name": "meta-llama/Llama-2-70b-hf", "type": "llama", "default_lr": 5e-5, "gradient_accumulation_steps": 4, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1, "load_in_8bit": True}
    ]
    
    # Define learning rates for hyperparameter sweep if enabled
    lr_values = []
    if args.run_lr_sweep:
        # For each model's default LR, create a range of values around it
        for model_info in models:
            default_lr = model_info["default_lr"]
            # Create a range: 1x, 2x, and 5x the default learning rate
            lr_values.extend([
                3e-5,
                4e-5,
                5e-5,
            ])
        # Remove duplicates and sort
        lr_values = sorted(list(set(lr_values)))
    else:
        # Just use the default learning rate for each model
        for model_info in models:
            lr_values.append(model_info["default_lr"])
    
    # Results tracking
    results = {}
    
    # Train each model with each bio field
    for model_info in models[::-1]:
        model_name = model_info["name"]
        model_results = {}
        
        for bio_field in bio_fields:
            print(f"\n{'='*80}")
            print(f"Training model: {model_name}")
            print(f"Bio field: {bio_field}")
            print(f"{'='*80}")
            
            model_type = model_info["type"]
            gradient_accumulation_steps = model_info["gradient_accumulation_steps"]
            per_device_train_batch_size = model_info["per_device_train_batch_size"]
            fp16 = model_info.get("fp16", False)
            gradient_checkpointing = model_info.get("gradient_checkpointing", False)
            eval_batch_size = model_info.get("eval_batch_size", 16)
            load_in_8bit = model_info.get("load_in_8bit", False)
            
            # Get short name for directories
            model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
            
            model_bio_results = []
            
            # Run with each learning rate in our sweep
            for learning_rate in (lr_values if args.run_lr_sweep else [model_info["default_lr"]]):
                # Create unique output directory for this model + bio field + learning rate combination
                lr_str = f"{learning_rate:.1e}".replace("-0", "-").replace("+0", "+")
                output_dir = os.path.join(
                    args.base_output_dir,
                    f"{model_short_name}-{bio_field}-lr{lr_str}"
                )
                
                print(f"\n{'-'*60}")
                print(f"Training {model_name} with:")
                print(f"Bio field: {bio_field}")
                print(f"Learning rate: {learning_rate}")
                print(f"Output directory: {output_dir}")
                print(f"{'-'*60}")

                if bio_field in ["bioS", "bioS_fullname"]:
                    num_train_epochs = 5 * args.num_train_epochs
                else:
                    num_train_epochs = args.num_train_epochs
                
                # Build command
                cmd = ["python",
                    "train_model.py",
                    "--model_name_or_path", model_name,
                    "--model_type", model_type,
                    "--learning_rate", str(learning_rate),
                    "--num_train_epochs", str(num_train_epochs),
                    "--per_device_train_batch_size", str(per_device_train_batch_size),
                    "--gradient_accumulation_steps", str(gradient_accumulation_steps),
                    "--max_samples", str(args.max_samples),
                    "--eval_samples", str(args.eval_samples),
                    "--eval_batch_size", str(eval_batch_size),
                    "--bio_field", bio_field,  # Use current bio field
                    "--output_dir", output_dir,
                    "--wandb_run_name", f"{model_short_name}-{bio_field}-lr{lr_str}",
                    "--freeze_embeddings",
                    "--mcq_percentage", str(args.mcq_percentage)
                ]
                
                if args.fp16 or fp16:
                    cmd.append("--fp16")
                
                if args.gradient_checkpointing or gradient_checkpointing:
                    cmd.append("--gradient_checkpointing")

                if load_in_8bit:
                    cmd.append("--load_in_8bit")
                
                if args.mcq_with_bios:
                    cmd.append("--mcq_with_bios")
                
                if args.shuffle_eval_choices:
                    cmd.append("--shuffle_eval_choices")
                
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
                    
                    # Store results with bio field
                    run_result["bio_field"] = bio_field
                    model_bio_results.append(run_result)
                    
                except subprocess.CalledProcessError as e:
                    print(f"Error training {model_name} with bio_field={bio_field}, lr={learning_rate}: {e}")
                    print(f"STDOUT: {e.stdout}")
                    print(f"STDERR: {e.stderr}")
                    
                    model_bio_results.append({
                        "bio_field": bio_field,
                        "learning_rate": learning_rate,
                        "status": "failed",
                        "error": str(e)
                    })
            
            # Store results for this bio field
            model_results[bio_field] = model_bio_results
        
        # Store all results for this model
        results[model_name] = model_results
    
    # Print summary of results grouped by model and bio field
    print("\n\n")
    print("=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    
    for model_name, model_results in results.items():
        print(f"\nModel: {model_name}")
        print("=" * 80)
        
        for bio_field, bio_results in model_results.items():
            print(f"\nBio Field: {bio_field}")
            print("-" * 80)
            
            # Sort results by best accuracy (descending)
            sorted_results = sorted(
                [r for r in bio_results if "best_accuracy" in r and r["best_accuracy"] is not None],
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
                print("No successful runs for this configuration")
                
                # Print failed runs
                failed_runs = [r for r in bio_results if "status" in r and r["status"] == "failed"]
                if failed_runs:
                    print("\nFailed runs:")
                    for failed in failed_runs:
                        print(f"  - LR {failed['learning_rate']}: {failed['error']}")
    
    # Save results to file with bio field information
    results_file = os.path.join(args.base_output_dir, "training_results.txt")
    with open(results_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write("TRAINING SUMMARY\n")
        f.write("=" * 100 + "\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"Model: {model_name}\n")
            f.write("=" * 80 + "\n")
            
            for bio_field, bio_results in model_results.items():
                f.write(f"\nBio Field: {bio_field}\n")
                f.write("-" * 80 + "\n")
                
                sorted_results = sorted(
                    [r for r in bio_results if "best_accuracy" in r and r["best_accuracy"] is not None],
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
                    f.write("No successful runs for this configuration\n")
                    
                    failed_runs = [r for r in bio_results if "status" in r and r["status"] == "failed"]
                    if failed_runs:
                        f.write("\nFailed runs:\n")
                        for failed in failed_runs:
                            f.write(f"  - LR {failed['learning_rate']}: {failed['error']}\n")
                
                f.write("\n")
            f.write("\n")
    
    print(f"\nDetailed training results saved to: {results_file}")
    print("\nTraining of all models complete!")

if __name__ == "__main__":
    main()
