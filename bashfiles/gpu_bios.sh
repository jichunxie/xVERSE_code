#!/bin/bash
#SBATCH -p biostat-gpu                     
#SBATCH -A biostat                           
#SBATCH --gres=gpu:1                        
#SBATCH -c 10                                
#SBATCH --mem=100G                        
#SBATCH -t 1:00:00                            
#SBATCH -J bios
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL                

# Load environment
source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code

# 4. Run Imputation Visualization
echo ">>> Running Task 4: Imputation Visualization"
stdbuf -oL -eL python -m fig5_imputation_generate_spatial.08_visualize_comparison_pearson

echo "=========================================================="
echo "All tests finished."
echo "=========================================================="


