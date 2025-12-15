#!/bin/bash
#SBATCH --job-name=Train_real
#SBATCH --output=Train_real%j.out
#SBATCH --error=Train_real%j.err
#SBATCH --nodes=1                   # Use all 4 nodes: 135,136,137,138
#SBATCH --ntasks=1                  # Total MPI ranks (32+16+16+8)
#SBATCH --cpus-per-task=32
#SBATCH --time=96:00:00             # Walltime limit
#SBATCH --mem=0                     # Use all available memory on each node
#SBATCH --partition=all		        # Partition to submit the job to

# Load modules (adjust these for your cluster)
module load python/3.10
module load cuda/12.1
module load pytorch/2.1.0

# Set OMP_NUM_THREADS to match requested CPUs-per-task for efficient CPU usage
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Print some info
echo "Job ID: ${SLURM_JOB_ID}"
echo "Running on: $(hostname)"
echo "Starting at: $(date)"

# Activate conda environment if needed
# source ~/.bashrc
# conda activate ML

echo "Starting Python training script: Train_real.py"
python Train_real.py

echo "Job finished at: $(date)"