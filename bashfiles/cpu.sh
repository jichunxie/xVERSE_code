#!/bin/bash
#SBATCH -p biostat                    
#SBATCH -A biostat                           
#SBATCH -c 30                                  
#SBATCH --mem=500G                        
#SBATCH -t 05:00:00                            
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

echo ">>> Building xVERSE index cache (train + val)"
stdbuf -oL -eL python -m main_energy.build_index_cache \
    --data-root "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor" \
    --train-cache-path "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/xverse_index_cache_train_all.npz" \
    --val-cache-path "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/xverse_index_cache_val_all.npz" \
    --filter-bad-cells
