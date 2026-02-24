#!/bin/bash
#SBATCH -p mesonet 
#SBATCH -N 1
#SBATCH -c 28
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=256G
#SBATCH --account=m25146
#SBATCH --job-name=generation
#SBATCH --output=outputs/%x_%j.out
#SBATCH --error=outputs/%x_%j.err

# --- ALL BASH COMMANDS GO BELOW THE SBATCH DIRECTIVES ---

name="generate"
outdir="outputs"

# Create the outputs directory if it doesn't exist so SLURM doesn't crash writing logs
mkdir -p $outdir

echo "Launching test for $name"

# Run your training script
python /home/peforcioli/B-PINNs/experiments/run_damped.py