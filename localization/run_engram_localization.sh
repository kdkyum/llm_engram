#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./job.out.%j
#SBATCH -e ./job.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J engram_loc
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000
#
#SBATCH --mail-type=none
#SBATCH --time=12:00:00

# Load required modules
module purge
module load cuda/12.6
module load python-waterboa/2024.06

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate engram

# Set environment variables
export WANDB_MODE=offline
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Create output directory if it doesn't exist
mkdir -p output_plots

# Run the engram localization script
# You can adjust these arguments based on your needs
python engram_localization.py \
    --pretrained "meta-llama/Llama-2-13b-hf" \
    --finetuned "/u/kdkyum/ptmp_link/workdir/llm_engram/model-output/meta-llama-Llama-2-13b-hf-bioS_multi5_permutes-evaldirect-ep5-bs1-samples20-lr5e-05-seed42/best_model" \
    --output_dir "output_plots" \
    --sample_size 100000 \
    --thresholds "0.01,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9" \
    --device "cpu"


python engram_localization.py \
    --pretrained "meta-llama/Llama-2-13b-hf" \
    --finetuned "/u/kdkyum/ptmp_link/workdir/llm_engram/model-output/meta-llama-Llama-2-13b-hf-bioS_multi5_permutes-evaldirect-ep5-bs1-samples20-lr3e-05-seed42/best_model" \
    --output_dir "output_plots" \
    --sample_size 100000 \
    --thresholds "0.01,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9" \
    --device "cpu"

python engram_localization.py \
    --pretrained "meta-llama/Llama-2-13b-hf" \
    --finetuned "/u/kdkyum/ptmp_link/workdir/llm_engram/model-output/meta-llama-Llama-2-13b-hf-bioS_multi5_permutes-evaldirect-ep5-bs1-samples20-lr7e-05-seed42/best_model" \
    --output_dir "output_plots" \
    --sample_size 100000 \
    --thresholds "0.01,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9" \
    --device "cpu"