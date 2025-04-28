#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
from datetime import datetime
import tempfile
import random
import string
import shutil
import numpy as np
import itertools  # Add this import for grid search

def parse_args():
    parser = argparse.ArgumentParser(description="Schedule LLM hyperparameter search jobs on GPU cluster")
    # Basic training parameters (can be overridden by model defaults)
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Base batch size (used for gradient accumulation calculation)")
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to use from dataset")
    parser.add_argument("--eval_samples", type=int, default=20, help="Number of samples to use for evaluation")
    parser.add_argument("--bio_field", type=str, default="bioS_multi5_permutes", help="Which bio field to use for training")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training (overrides model default if set)")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing (overrides model default if set)")
    parser.add_argument("--base_output_dir", type=str, default="./model-output-hparam-search", help="Base directory to save models for this search")
    parser.add_argument("--mcq_percentage", type=int, default=0, help="Percentage of MCQ evaluation examples to include in training data (0-100)")
    parser.add_argument("--mcq_with_bios", action="store_true", help="Include bioS text with MCQ examples (if not set, only MCQ text is used)")
    parser.add_argument("--shuffle_eval_choices", action="store_true", help="Shuffle MCQ choices during evaluation to test for overfitting")
    
    # SLURM/Job Management Parameters
    parser.add_argument("--jobs_dir", type=str, default="./slurm_hparam_jobs", help="Directory to store SLURM job scripts for hyperparameter search")
    parser.add_argument("--time_limit", type=str, default="12:00:00", help="Time limit for each job (HH:MM:SS)")
    parser.add_argument("--max_parallel_jobs", type=int, default=10, help="Maximum number of hyperparameter search jobs to run in parallel")
    parser.add_argument("--memory_per_gpu", type=int, default=125000, help="Memory in MB per GPU (adjust if needed)")
    parser.add_argument("--wait_time", type=int, default=60, help="Wait time in seconds between job status checks")
    parser.add_argument("--mail_type", type=str, default="NONE", choices=["NONE", "BEGIN", "END", "FAIL", "ALL"], help="When to send email notifications")
    parser.add_argument("--mail_user", type=str, default=None, help="Email address for notifications")

    # Hyperparameter Search Specific Arguments
    parser.add_argument("--search_model_name", type=str, default="meta-llama/Llama-2-13b-hf", help="Model name for hyperparameter search")
    # Modify parameters for grid search
    parser.add_argument("--learning_rates", type=str, default="1e-5,2e-5,5e-5,1e-4", help="Comma-separated list of learning rates for grid search")
    parser.add_argument("--weight_decays", type=str, default="0.0,1e-5,1e-4,1e-3,1e-2", help="Comma-separated list of weight decay values for grid search")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for reproducibility of sampling (jobs themselves might use different seeds)")

    return parser.parse_args()

def random_string(length=8):
    """Generate a random string for unique job names"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_running_jobs():
    """Get list of currently running jobs for the user"""
    try:
        result = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], 
                               capture_output=True, text=True, check=True)
        job_ids = [job_id.strip() for job_id in result.stdout.strip().split('\n') if job_id.strip()]
        return job_ids
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not query SLURM jobs ({e}). Assuming no jobs are running.")
        return []

def create_slurm_script(args, model_info, bio_field, learning_rate, weight_decay, job_id):
    """Create a SLURM job script for a specific hyperparameter trial"""
    model_name = model_info["name"]
    model_type = model_info["type"]
    # Use model defaults unless overridden by general args
    gradient_accumulation_steps = model_info.get("gradient_accumulation_steps", args.batch_size)
    per_device_train_batch_size = model_info.get("per_device_train_batch_size", 1)
    fp16 = args.fp16 or model_info.get("fp16", False)
    gradient_checkpointing = args.gradient_checkpointing or model_info.get("gradient_checkpointing", False)
    eval_batch_size = model_info.get("eval_batch_size", 16)
    load_in_8bit = model_info.get("load_in_8bit", False) # Assuming this might be a model-specific option
    
    # Get short name for directories and job naming
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Format learning rate and weight decay strings for names/paths
    lr_str = f"{learning_rate:.1e}".replace("-0", "-").replace("+0", "+")
    wd_str = f"{weight_decay:.1e}".replace("-0", "-").replace("+0", "+")
    
    # Create unique job name including hyperparameters
    job_name = f"hps-{model_short_name[:8]}-lr{lr_str}-wd{wd_str}-{job_id}"
    
    # Create logs directory within the specific jobs_dir for this script
    logs_dir = os.path.join(args.jobs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define number of GPUs, CPUs, and memory (using Raven specs for 4 GPUs)
    # These could potentially be adjusted based on model_info if needed
    num_gpus = 4
    cpus_per_task = 72
    mem = 500000 # 500 GB for a full node
    
    # Adjust epochs based on bio field (same logic as schedule_training.py)
    if bio_field in ["bioS", "bioS_fullname"]:
        num_train_epochs = 5 * args.num_train_epochs
    else:
        num_train_epochs = args.num_train_epochs
    
    # Build python command for train_model.py, passing the specific LR and WD
    python_cmd = ["python",
        "train_model.py",
        "--model_name_or_path", model_name,
        "--model_type", model_type,
        "--learning_rate", str(learning_rate), # Pass sampled LR
        "--weight_decay", str(weight_decay),   # Pass sampled WD
        "--num_train_epochs", str(num_train_epochs),
        "--per_device_train_batch_size", str(per_device_train_batch_size),
        "--gradient_accumulation_steps", str(gradient_accumulation_steps),
        "--max_samples", str(args.max_samples),
        "--eval_samples", str(args.eval_samples),
        "--eval_batch_size", str(eval_batch_size),
        "--bio_field", bio_field,
        "--output_dir", args.base_output_dir, # Use the dedicated output dir for HPO
        "--wandb_project", f"llm-engram-hparam-search", # Suggest separate W&B project
        "--wandb_run_name", f"{model_short_name}-{bio_field}-lr{lr_str}-wd{wd_str}", # Include WD in run name
        "--freeze_embeddings", # Assuming this is standard for your runs
        "--offline", # Run wandb offline as per instructions
        "--debug",
        "--mcq_percentage", str(args.mcq_percentage)
        # Add other relevant flags from args
    ]
    
    if fp16:
        python_cmd.append("--fp16")
    if gradient_checkpointing:
        python_cmd.append("--gradient_checkpointing")
    if load_in_8bit:
        python_cmd.append("--load_in_8bit")
    if args.mcq_with_bios:
        python_cmd.append("--mcq_with_bios")
    if args.shuffle_eval_choices:
        python_cmd.append("--shuffle_eval_choices")
    
    # Format the Python command as a string
    formatted_python_cmd = " ".join(python_cmd)
    
    # Create SLURM script content using Raven template
    slurm_script = f"""#!/bin/bash -l
#SBATCH -o {logs_dir}/job.out.%j
#SBATCH -e {logs_dir}/job.err.%j
#SBATCH -D ./
#SBATCH -J {job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:{num_gpus}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
"""
    
    # Add mail options if specified
    if args.mail_type != "NONE" and args.mail_user:
        slurm_script += f"#SBATCH --mail-type={args.mail_type}\n"
        slurm_script += f"#SBATCH --mail-user={args.mail_user}\n"
    
    slurm_script += f"""#SBATCH --time={args.time_limit}

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate engram

export WANDB_MODE=offline
export OMP_NUM_THREADS=${{SLURM_CPUS_PER_TASK:-1}} # Set OMP_NUM_THREADS

# Debugging info
echo "Starting job $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Using model: {model_name}"
echo "Learning Rate: {learning_rate}"
echo "Weight Decay: {weight_decay}"

# Run the model training
{formatted_python_cmd}

# Get exit status
status=$?
echo "Job $SLURM_JOB_ID finished with status: $status"

# Save exit status to file for the monitoring script
echo $status > {logs_dir}/exit_status.{job_name}

exit $status
"""
    
    return slurm_script, job_name

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Create jobs directory and logs subdirectory
    os.makedirs(os.path.join(args.jobs_dir, "logs"), exist_ok=True)
    os.makedirs(args.base_output_dir, exist_ok=True)
    
    # Define all models (needed to find the one being searched)
    # This should match the list in schedule_training.py or be loaded from a common source
    all_models = [
        {"name": "gpt2-xl", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 1},
        {"name": "EleutherAI/gpt-j-6B", "type": "gpt-j", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-7b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-3.1-8B", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-7B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-13b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-13B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
    ]

    # Find the model info for the search
    search_model_info = next((m for m in all_models if m['name'] == args.search_model_name), None)
    if not search_model_info:
        print(f"Error: Model '{args.search_model_name}' not found in the defined models list.")
        return

    print(f"\n{'='*80}")
    print(f"PREPARING GRID SEARCH for {args.search_model_name}")
    print(f"Model Info: {search_model_info}")
    print(f"Bio Field: {args.bio_field}")
    print(f"{'='*80}")

    # Parse learning rates and weight decays from string to float values
    learning_rates = [float(lr.strip()) for lr in args.learning_rates.split(",")]
    weight_decays = [float(wd.strip()) for wd in args.weight_decays.split(",")]
    
    # Generate all combinations for grid search
    param_combinations = list(itertools.product(learning_rates, weight_decays))
    
    print(f"Grid Search Configuration:")
    print(f"  - Learning rates: {learning_rates}")
    print(f"  - Weight decays: {weight_decays}")
    print(f"  - Total combinations: {len(param_combinations)}")
    print(f"  - Output Directory: {args.base_output_dir}")
    print(f"  - Job Scripts Directory: {args.jobs_dir}")

    job_queue = []

    print(f"\nPreparing {len(param_combinations)} job scripts...")
    for i, (learning_rate, weight_decay) in enumerate(param_combinations):
        job_id = random_string() # Unique ID for script naming

        # Create SLURM script
        slurm_script, job_name = create_slurm_script(args, search_model_info, args.bio_field, learning_rate, weight_decay, job_id)

        # Write script to file
        script_path = os.path.join(args.jobs_dir, f"{job_name}.sh")
        with open(script_path, "w") as f:
            f.write(slurm_script)
        os.chmod(script_path, 0o755) # Make script executable

        # Add to job queue
        job_queue.append({
            "model_name": search_model_info['name'],
            "bio_field": args.bio_field,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "script_path": script_path,
            "job_name": job_name,
            "status": "pending",
            "job_id": None # SLURM job ID assigned after submission
        })
        print(f"  - Trial {i+1:02d}: LR={learning_rate:.3e}, WD={weight_decay:.3e} -> {script_path}")

    print(f"\n{'='*80}")
    print(f"PREPARED {len(job_queue)} JOBS")
    print(f"{'='*80}")
    print(f"Maximum parallel jobs: {args.max_parallel_jobs}")
    print(f"Job scripts directory: {args.jobs_dir}")
    print(f"Logs directory: {os.path.join(args.jobs_dir, 'logs')}")
    
    confirm = input("\nDo you want to submit these jobs to SLURM? (yes/no): ")
    if confirm.lower() not in ["yes", "y"]:
        print("Job submission cancelled.")
        # Optionally clean up generated scripts
        # cleanup = input("Clean up generated scripts? (yes/no): ")
        # if cleanup.lower() in ["yes", "y"]:
        #     shutil.rmtree(args.jobs_dir)
        #     print(f"Removed directory: {args.jobs_dir}")
        return
    
    running_jobs = []
    completed_jobs = []
    failed_jobs = []
    
    print(f"\n{'='*80}")
    print("SUBMITTING AND MONITORING JOBS")
    print(f"{'='*80}")
    
    try:
        while job_queue or running_jobs:
            current_running_job_ids = get_running_jobs()
            
            # --- Update status of running jobs --- 
            for job in running_jobs[:]: # Iterate over a copy
                if job["job_id"] not in current_running_job_ids:
                    # Job finished, check status
                    exit_status_file = os.path.join(args.jobs_dir, "logs", f"exit_status.{job['job_name']}")
                    lr = job['learning_rate']
                    wd = job['weight_decay']
                    status_msg = f"{job['model_name']}, LR: {lr:.3e}, WD: {wd:.3e} (Job ID: {job['job_id']})"
                    
                    job_status = "unknown"
                    try:
                        if os.path.exists(exit_status_file):
                            with open(exit_status_file, "r") as f:
                                exit_status = int(f.read().strip())
                            if exit_status == 0:
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job completed: {status_msg}")
                                job_status = "completed"
                            else:
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job failed (status {exit_status}): {status_msg}")
                                job_status = "failed"
                        else:
                            # Fallback if exit status file is missing (less reliable)
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job finished (exit status file missing): {status_msg}. Assuming failure.")
                            job_status = "failed"
                            
                    except ValueError:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job finished (invalid exit status): {status_msg}. Assuming failure.")
                        job_status = "failed"
                    except Exception as e:
                         print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error checking job status for {status_msg}: {e}. Assuming failure.")
                         job_status = "failed"

                    # Update job lists
                    job["status"] = job_status
                    if job_status == "completed":
                        completed_jobs.append(job)
                    else: # failed or unknown
                        failed_jobs.append(job)
                    
                    running_jobs.remove(job)

            # --- Submit new jobs --- 
            while job_queue and len(running_jobs) < args.max_parallel_jobs:
                job = job_queue.pop(0)
                lr = job['learning_rate']
                wd = job['weight_decay']
                submit_msg = f"{job['model_name']}, LR: {lr:.3e}, WD: {wd:.3e}"
                
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Submitting job: {submit_msg}")
                try:
                    result = subprocess.run(["sbatch", job["script_path"]], capture_output=True, text=True, check=True)
                    if "Submitted batch job" in result.stdout:
                        job_id = result.stdout.strip().split()[-1]
                        job["job_id"] = job_id
                        job["status"] = "running"
                        running_jobs.append(job)
                        print(f"  - Job submitted with ID: {job_id}")
                    else:
                        # This case might not happen if check=True catches the error
                        print(f"  - Failed to submit job (unexpected output): {submit_msg}\n{result.stdout}\n{result.stderr}")
                        job["status"] = "failed"
                        failed_jobs.append(job)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    print(f"  - Failed to submit job: {submit_msg}\nError: {e}")
                    job["status"] = "failed"
                    failed_jobs.append(job)
                except Exception as e:
                    print(f"  - An unexpected error occurred during submission for {submit_msg}: {e}")
                    job["status"] = "failed"
                    failed_jobs.append(job)
            
            # --- Print status update --- 
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Status: {len(running_jobs)} running, {len(job_queue)} pending, {len(completed_jobs)} completed, {len(failed_jobs)} failed")
            
            # --- Wait before next check --- 
            if job_queue or running_jobs:
                time.sleep(args.wait_time)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted. Stopping job submission and monitoring...")
        
        # Ask user if they want to cancel running SLURM jobs
        if running_jobs:
            cancel = input(f"Do you want to cancel the {len(running_jobs)} running SLURM jobs? (yes/no): ")
            if cancel.lower() in ["yes", "y"]:
                cancelled_count = 0
                for job in running_jobs:
                    if job["job_id"]:
                        print(f"Cancelling job {job['job_id']} ({job['job_name']})...")
                        try:
                            subprocess.run(["scancel", job["job_id"]], check=True)
                            cancelled_count += 1
                        except (subprocess.CalledProcessError, FileNotFoundError) as e:
                            print(f"  - Failed to cancel job {job['job_id']}: {e}")
                print(f"Attempted to cancel {cancelled_count} jobs.")
            else:
                print("Running jobs were not cancelled.")
        else:
            print("No running jobs to cancel.")
    
    # --- Final Summary --- 
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SEARCH JOB SUMMARY")
    print(f"{'='*80}")
    total_submitted = len(completed_jobs) + len(failed_jobs) + len(running_jobs)
    print(f"Total trials submitted: {total_submitted}")
    print(f"Completed successfully: {len(completed_jobs)}")
    print(f"Failed or unknown status: {len(failed_jobs)}")
    print(f"Still running (if not cancelled): {len(running_jobs)}")
    print(f"Not submitted (pending queue): {len(job_queue)}")
    
    # List failed jobs
    if failed_jobs:
        print("\nFailed/Unknown Status Jobs:")
        for job in failed_jobs:
            lr = job['learning_rate']
            wd = job['weight_decay']
            print(f"  - Script: {os.path.basename(job['script_path'])}, LR: {lr:.3e}, WD: {wd:.3e}, SLURM ID: {job.get('job_id', 'N/A')}")
    
    # Ask for cleanup
    if not running_jobs and not job_queue:
        cleanup = input(f"\nDo you want to clean up the job script directory '{args.jobs_dir}'? (yes/no): ")
        if cleanup.lower() in ["yes", "y"]:
            try:
                shutil.rmtree(args.jobs_dir)
                print(f"Removed directory: {args.jobs_dir}")
            except OSError as e:
                print(f"Error removing directory {args.jobs_dir}: {e}")
        else:
            print(f"Job scripts and logs preserved in {args.jobs_dir}")
    else:
        print(f"\nJob scripts and logs preserved in {args.jobs_dir}")

if __name__ == "__main__":
    main()
