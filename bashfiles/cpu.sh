#!/bin/bash
#SBATCH -p biostat                    
#SBATCH -A biostat                           
#SBATCH -c 20                                  
#SBATCH --mem=200G                        
#SBATCH -t 20:00:00                            
#SBATCH -J build_ct_text
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
CELLTYPE_CSV="/hpc/group/xielab/xj58/general/cellxgene_cell_type_id2name.csv"
OPENAI_KEY_PATH="/hpc/group/xielab/xj58/general/openai_xielab.txt"
TEXT_EMB_PREFIX="/hpc/group/xielab/xj58/general/cellxgene_cell_type_text"
TEXT_EMB_NPZ="${TEXT_EMB_PREFIX}_embeddings.npz"
COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"

echo ">>> Building cell-type text embeddings + updating compiled celltype labels"
echo ">>> DATA_ROOT=${DATA_ROOT}"
echo ">>> SUMMARY_CSV=${SUMMARY_CSV}"
echo ">>> GENE_IDS_PATH=${GENE_IDS_PATH}"
echo ">>> CELLTYPE_CSV=${CELLTYPE_CSV}"
echo ">>> TEXT_EMB_NPZ=${TEXT_EMB_NPZ}"
echo ">>> COMPILED_ROOT=${COMPILED_ROOT}"

if [ ! -f "${TEXT_EMB_NPZ}" ]; then
  echo ">>> Building cell type language embeddings"
  stdbuf -oL -eL python -u -m main_mfa.build_celltype_text_embeddings \
      --cell-type-csv "${CELLTYPE_CSV}" \
      --api-key-path "${OPENAI_KEY_PATH}" \
      --output-prefix "${TEXT_EMB_PREFIX}"
else
  echo ">>> Reusing existing cell type language embeddings: ${TEXT_EMB_NPZ}"
fi

echo ">>> Updating celltype_id.npy in existing compiled dataset"
stdbuf -oL -eL python -u -m main_mfa.update_compiled_celltype_ids \
    --compiled-root "${COMPILED_ROOT}" \
    --cell-type-csv "${CELLTYPE_CSV}" \
    --splits train,val \
    --backup
