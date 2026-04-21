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

COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"
CKPT_PATH="/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue_h200/best_model.pth"
CELLTYPE_CSV="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/cellxgene_cell_type_mapped.csv"

echo ">>> Running Task: Analyze dominant GMM component by cell type"
echo ">>> CKPT_PATH=${CKPT_PATH}"
echo ">>> COMPILED_ROOT=${COMPILED_ROOT}"
stdbuf -oL -eL python -m main_energy.analyze_gmm_component_by_celltype \
    --ckpt-path "${CKPT_PATH}" \
    --compiled-dataset-root "${COMPILED_ROOT}" \
    --split val \
    --batch-size 1024 \
    --num-workers 4 \
    --cell-type-csv "${CELLTYPE_CSV}"

echo "=========================================================="
echo "Analysis finished."
echo "=========================================================="

