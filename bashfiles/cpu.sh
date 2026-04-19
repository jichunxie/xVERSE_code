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

DATA_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor"
SUMMARY_CSV="${DATA_ROOT}/pantissue_full_updated.csv"
GENE_IDS_PATH="${DATA_ROOT}/ensg_keys_high_quality.txt"
CELLTYPE_CSV="${DATA_ROOT}/cellxgene_cell_type_mapped.csv"
OUT_DIR="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"

echo ">>> Building training dataset (xverse_train_v1)"
echo ">>> DATA_ROOT=${DATA_ROOT}"
echo ">>> SUMMARY_CSV=${SUMMARY_CSV}"
echo ">>> GENE_IDS_PATH=${GENE_IDS_PATH}"
echo ">>> CELLTYPE_CSV=${CELLTYPE_CSV}"
echo ">>> OUT_DIR=${OUT_DIR}"

stdbuf -oL -eL python -m main_energy.build_train_dataset_v1 \
    --summary-csv "${SUMMARY_CSV}" \
    --gene-ids-path "${GENE_IDS_PATH}" \
    --cell-type-csv "${CELLTYPE_CSV}" \
    --split both \
    --target-cells-per-shard 200000 \
    --out-dir "${OUT_DIR}" \
    --filter-bad-cells
