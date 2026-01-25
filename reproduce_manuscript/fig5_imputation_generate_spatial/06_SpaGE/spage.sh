#!/bin/bash
#SBATCH --job-name=spage4000
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_job_output_%j.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_job_error_%j.txt
#SBATCH --partition=chsi
#SBATCH --account=chsi
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL  

echo "Job started at $(date)"
echo "Running on node $(hostname)"

source activate SpaRest

cd /hpc/group/xielab/xj58/sparest_code


# Run your application, for example a Python script:
python -u benchmark/SpaGE/spaGE_250.py

echo "Job finished at $(date)"