import os
import argparse
import subprocess
import time
from datetime import datetime
import tempfile
import random
import string
import shutil

# Import optim_configs
from helpers.optim_configs import optim_configs

def parse_args():
    parser = argparse.ArgumentParser(description="Schedule LLM training jobs on GPU cluster")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=20, help="Maximum number of samples to use from dataset")
    parser.add_argument("--eval_samples", type=int, default=20, help="Number of samples to use for evaluation")
    parser.add_argument("--bio_field", type=str, default="bioS", help="Which bio field to use for training")
    parser.add_argument("--run_lr_sweep", action="store_true", help="Run learning rate sweep for each model")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--base_output_dir", type=str, default="./model-output", help="Base directory to save models")
    parser.add_argument("--mcq_percentage", type=int, default=0, help="Percentage of MCQ evaluation examples to include in training data (0-100)")
    parser.add_argument("--mcq_with_bios", action="store_true", help="Include bioS text with MCQ examples (if not set, only MCQ text is used)")
    parser.add_argument("--shuffle_eval_choices", action="store_true", help="Shuffle MCQ choices during evaluation to test for overfitting")
    parser.add_argument("--jobs_dir", type=str, default="./slurm_jobs", help="Directory to store SLURM job scripts")
    parser.add_argument("--time_limit", type=str, default="12:00:00", help="Time limit for each job (HH:MM:SS)")
    parser.add_argument("--max_parallel_jobs", type=int, default=40, help="Maximum number of jobs to run in parallel")
    parser.add_argument("--memory_per_gpu", type=int, default=125000, help="Memory in MB per GPU")
    parser.add_argument("--wait_time", type=int, default=60, help="Wait time in seconds between job status checks")
    parser.add_argument("--mail_type", type=str, default="FAIL", choices=["NONE", "BEGIN", "END", "FAIL", "ALL"], help="When to send email notifications")
    parser.add_argument("--mail_user", type=str, default=None, help="Email address for notifications")
    return parser.parse_args()

def random_string(length=8):
    """Generate a random string for unique job names"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_running_jobs():
    """Get list of currently running jobs for the user"""
    result = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-o", "%i"], 
                           capture_output=True, text=True)
    job_ids = [job_id.strip() for job_id in result.stdout.strip().split('\n') if job_id.strip()]
    return job_ids

def create_slurm_script(args, model_info, bio_field, learning_rate, job_id):
    """Create a SLURM job script for the given parameters"""
    model_name = model_info["name"]
    model_type = model_info["type"]
    gradient_accumulation_steps = model_info["gradient_accumulation_steps"]
    per_device_train_batch_size = model_info["per_device_train_batch_size"]
    fp16 = model_info.get("fp16", False) or args.fp16
    gradient_checkpointing = model_info.get("gradient_checkpointing", False) or args.gradient_checkpointing
    eval_batch_size = model_info.get("eval_batch_size", 16)
    load_in_8bit = model_info.get("load_in_8bit", False)
    
    # Get short name for directories and job naming
    model_short_name = model_name.split("/")[-1] if "/" in model_name else model_name
    
    # Format learning rate string
    lr_str = f"{learning_rate:.1e}".replace("-0", "-").replace("+0", "+")
    
    # Create unique output directory
    output_dir = args.base_output_dir
    
    # Create job name
    job_name = f"engram-{model_short_name[:10]}-lr{lr_str}-{job_id}"
    
    # Create logs directory
    logs_dir = os.path.join(args.jobs_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Define number of GPUs, CPUs, and memory based on model size
    num_gpus = 4
    cpus_per_task = 72
    mem = 500000
    
    # Adjust epochs based on bio field
    if bio_field in ["bioS", "bioS_fullname"]:
        num_train_epochs = 5 * args.num_train_epochs
    else:
        num_train_epochs = args.num_train_epochs
    
    # Build python command for training
    python_cmd = ["python",
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
        "--bio_field", bio_field,
        "--output_dir", output_dir,
        "--wandb_run_name", f"{model_short_name}-{bio_field}-lr{lr_str}",
        "--freeze_embeddings",
        "--offline",
        "--mcq_percentage", str(args.mcq_percentage)
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
    
    # Create SLURM script content
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
    
    slurm_script += f"""#SBATCH --time={args.time_limit}

module purge
module load cuda/12.6
module load python-waterboa/2024.06

eval "$(conda shell.bash hook)"
conda activate engram
"""
    
    slurm_script += f"""
# Run the model training
{formatted_python_cmd}

# Get exit status
status=$?

# Save exit status to file for the monitoring script
echo $status > {logs_dir}/exit_status.{job_name}

exit $status
"""
    
    return slurm_script, job_name

def main():
    args = parse_args()
    
    # Create jobs directory
    os.makedirs(args.jobs_dir, exist_ok=True)
    
    # Define all possible bio fields
    bio_fields = ["bioS_multi5_permutes"]  # You can modify this as needed
    
    # Define models to train
    all_models = [
        {"name": "gpt2-xl", "type": "gpt2", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "eval_batch_size": 1},
        {"name": "EleutherAI/gpt-j-6B", "type": "gpt-j", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-7b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-3.1-8B", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-7B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "meta-llama/Llama-2-13b-hf", "type": "llama", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
        {"name": "allenai/OLMo-2-1124-13B", "type": "olmo", "default_lr": 2e-5, "gradient_accumulation_steps": args.batch_size, "per_device_train_batch_size": 1, "fp16": True, "gradient_checkpointing": True, "eval_batch_size": 1},
    ]
    
    job_queue = []
    
    # Define learning rates for hyperparameter sweep if enabled
    lr_values = []
    if args.run_lr_sweep:
        # For each model's default LR, create a range of values
        lr_values.extend([3e-5, 5e-5, 7e-5, 1e-4])
        # Remove duplicates and sort
        lr_values = sorted(list(set(lr_values)))
    else:
        # Just use the default learning rate for each model
        for model_info in all_models:
            lr_values.append(model_info["default_lr"])
    
    # Create job configurations for all combinations
    for model_info in all_models[::-1]:
        model_name = model_info["name"]
        
        for bio_field in bio_fields:
            print(f"\nPreparing jobs for model: {model_name}, bio field: {bio_field}")
            
            # For each learning rate
            for learning_rate in (lr_values if args.run_lr_sweep else [model_info["default_lr"]]):
                # Generate a unique job ID
                job_id = random_string()
                
                # Create SLURM script
                slurm_script, job_name = create_slurm_script(args, model_info, bio_field, learning_rate, job_id)
                
                # Write script to file
                script_path = os.path.join(args.jobs_dir, f"{job_name}.sh")
                with open(script_path, "w") as f:
                    f.write(slurm_script)
                
                # Add to job queue
                job_queue.append({
                    "model_name": model_name,
                    "bio_field": bio_field,
                    "learning_rate": learning_rate,
                    "script_path": script_path,
                    "job_name": job_name,
                    "status": "pending",
                    "job_id": None
                })
                
                print(f"  - Created job script: {script_path}")

    print(f"\n{'='*80}")
    print(f"PREPARED {len(job_queue)} JOBS")
    print(f"{'='*80}")
    print(f"Maximum parallel jobs: {args.max_parallel_jobs}")
    print(f"Job scripts directory: {args.jobs_dir}")
    print(f"Logs directory: {os.path.join(args.jobs_dir, 'logs')}")
    
    confirm = input("\nDo you want to submit these jobs to SLURM? (yes/no): ")
    if confirm.lower() not in ["yes", "y"]:
        print("Job submission cancelled.")
        return
    
    running_jobs = []
    completed_jobs = []
    failed_jobs = []
    
    print(f"\n{'='*80}")
    print("SUBMITTING JOBS")
    print(f"{'='*80}")
    
    try:
        while job_queue or running_jobs:
            current_running_job_ids = get_running_jobs()
            
            # Update status of running jobs
            for job in running_jobs[:]:
                if job["job_id"] not in current_running_job_ids:
                    # Check exit status file first
                    exit_status_file = os.path.join(args.jobs_dir, "logs", f"exit_status.{job['job_name']}")
                    lr = job['learning_rate']
                    if os.path.exists(exit_status_file):
                        with open(exit_status_file, "r") as f:
                            try:
                                exit_status = int(f.read().strip())
                                if exit_status == 0:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job completed: {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                                    job["status"] = "completed"
                                    completed_jobs.append(job)
                                else:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job failed (status {exit_status}): {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                                    job["status"] = "failed"
                                    failed_jobs.append(job)
                            except ValueError:
                                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job status unknown: {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                                job["status"] = "failed"
                                failed_jobs.append(job)
                    else:
                        # Fallback: Check output file for success message (less reliable)
                        output_file = os.path.join(args.jobs_dir, "logs", f"job.out.{job['job_id']}")
                        if os.path.exists(output_file):
                            with open(output_file, "r") as f:
                                output = f.read()
                                # Adjust this success message if your training script outputs something different
                                if "Training of all models complete!" in output:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job completed: {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                                    job["status"] = "completed"
                                    completed_jobs.append(job)
                                else:
                                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job likely failed (no success msg): {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                                    job["status"] = "failed"
                                    failed_jobs.append(job)
                        else:
                            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Job likely failed (no output): {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                            job["status"] = "failed"
                            failed_jobs.append(job)
                    
                    # Remove from running jobs list
                    running_jobs.remove(job)

            # Submit new jobs if there's room
            while job_queue and len(running_jobs) < args.max_parallel_jobs:
                job = job_queue.pop(0)
                lr = job['learning_rate']
                # Submit job to SLURM
                print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Submitting job: {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
                result = subprocess.run(["sbatch", job["script_path"]], capture_output=True, text=True)
                
                if result.returncode == 0 and "Submitted batch job" in result.stdout:
                    job_id = result.stdout.strip().split()[-1]
                    job["job_id"] = job_id
                    job["status"] = "running"
                    running_jobs.append(job)
                    print(f"  - Job submitted with ID: {job_id}")
                else:
                    print(f"  - Failed to submit job: {result.stderr}")
                    job["status"] = "failed"
                    failed_jobs.append(job)
            
            # Print status update
            print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Status: {len(running_jobs)} running, {len(job_queue)} pending, {len(completed_jobs)} completed, {len(failed_jobs)} failed")
            
            # Wait before checking again
            if job_queue or running_jobs:
                time.sleep(args.wait_time)
    
    except KeyboardInterrupt:
        print("\nProcess interrupted. Stopping job submission...")
        
        # Ask user if they want to cancel running jobs
        cancel = input("Do you want to cancel running jobs? (yes/no): ")
        if cancel.lower() in ["yes", "y"]:
            for job in running_jobs:
                if job["job_id"]:
                    print(f"Cancelling job {job['job_id']}...")
                    subprocess.run(["scancel", job["job_id"]])
    
    # Final summary
    print(f"\n{'='*80}")
    print("JOB SUMMARY")
    print(f"{'='*80}")
    print(f"Total jobs: {len(completed_jobs) + len(failed_jobs) + len(running_jobs) + len(job_queue)}")
    print(f"Completed: {len(completed_jobs)}")
    print(f"Failed: {len(failed_jobs)}")
    print(f"Still running: {len(running_jobs)}")
    print(f"Pending: {len(job_queue)}")
    
    # List failed jobs
    if failed_jobs:
        print("\nFailed jobs:")
        for job in failed_jobs:
            lr = job['learning_rate']
            print(f"  - {job['model_name']}, {job['bio_field']}, LR: {lr:.2e}")
    
    # Ask for cleanup if all jobs are finished
    if not running_jobs and not job_queue:
        cleanup = input("\nDo you want to clean up job scripts? (yes/no): ")
        if cleanup.lower() in ["yes", "y"]:
            for job in completed_jobs + failed_jobs:
                if os.path.exists(job["script_path"]):
                    os.remove(job["script_path"])
            print("Job scripts cleaned up. Logs have been preserved.")

if __name__ == "__main__":
    main()
