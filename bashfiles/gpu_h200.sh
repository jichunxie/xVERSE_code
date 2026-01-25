#!/bin/bash
#SBATCH -p h200ea                            
#SBATCH -A h200ea                                
#SBATCH --gres=gpu:h200:5
#SBATCH -c 20                                 
#SBATCH --mem=1450G                        
#SBATCH -t 70:00:00                            
#SBATCH -J h200
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --exclude=dcc-h200-gpu-05                 

# Load environment
source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code

echo ">>> Running Task 0: Pretrain Pan-Tissue Model"
stdbuf -oL -eL python -m main.train_pantissue \
    --hidden-dim 64 \
    --result-dir "/hpc/group/xielab/xj58/pretrain_model_celltype/pantissue_model_64" \
    --num-epochs 50
