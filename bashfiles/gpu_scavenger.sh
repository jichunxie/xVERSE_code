#!/bin/bash
#SBATCH -p scavenger-gpu                
#SBATCH -A xielab                 
#SBATCH --gres=gpu:1
#SBATCH -c 10                             
#SBATCH --mem=100G                        
#SBATCH -t 7:00:00                            
#SBATCH -J scavenger-gpu
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL                

# Load environment
source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code

DATA_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor"
COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"
RESULT_DIR="/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue3"

CKPT_PATH="${RESULT_DIR}/best_model.pth"
FIG2_LIVER_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
FIG2_BRAIN_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
FIG2_GENE_IDS="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
FIG2_OUT_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_current"
FIG2_EVAL_DIR="${FIG2_OUT_DIR}/evaluation_scib_full"
FIG2_OLD_EVAL_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/evaluation"

echo ">>> Running Task: Extract GMVAE embeddings on fig2 donor h5ad files"
python reproduce_manuscript/fig2_biology_signal_gmmvae_current/02_extract_gmmvae_embedding.py \
    --ckpt "${CKPT_PATH}" \
    --gene-ids-path "${FIG2_GENE_IDS}" \
    --liver-dir "${FIG2_LIVER_DIR}" \
    --brain-dir "${FIG2_BRAIN_DIR}" \
    --output-dir "${FIG2_OUT_DIR}" \
    --embedding-key xVerse_gmmvae

echo ">>> Running Task: Evaluate FMs + GMVAE with full scIB metrics"
python reproduce_manuscript/fig2_biology_signal_gmmvae_current/03_evaluate_scib_full.py \
    --liver-dir "${FIG2_LIVER_DIR}" \
    --brain-dir "${FIG2_BRAIN_DIR}" \
    --output-dir "${FIG2_EVAL_DIR}" \
    --old-eval-dir "${FIG2_OLD_EVAL_DIR}" \
    --gmm-key xVerse_gmmvae


# python main_energy/diagnose_ckpt_val.py \
#   --ckpt /hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue2/last_model.pth \
#   --compiled-dataset-root /hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all \
#   --val-num-workers 4
