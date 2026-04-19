#!/bin/bash
#SBATCH -p biostat                    
#SBATCH -A biostat                           
#SBATCH -c 20                                  
#SBATCH --mem=200G                        
#SBATCH -t 10:00:00                            
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

echo ">>> Compiling NPZ dataset to mask-dictionary shard format"
stdbuf -oL -eL python -m main_energy.compile_maskdict_dataset \
    --data-root "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor" \
    --cell-type-csv "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/cellxgene_cell_type_mapped.csv" \
    --split both \
    --out-dir "/hpc/group/xielab/xj58/xVerseAtlas/compiled_maskdict_all" \
    --filter-bad-cells
