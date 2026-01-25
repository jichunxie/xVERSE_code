#!/bin/bash
#SBATCH -p biostat                    
#SBATCH -A biostat                           
#SBATCH -c 30                                  
#SBATCH --mem=200G                        
#SBATCH -t 01:00:00                            
#SBATCH -J bios_cpu
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL                

# Load environment
source ~/.bashrc
conda activate SpaRest

# Change to working directory
cd /hpc/group/xielab/xj58/xVERSE_code

echo ">>> Running Task 4: Imputation Visualization"
stdbuf -oL -eL python -m fig5_imputation_generate_spatial.08_visualize_comparison_pearson