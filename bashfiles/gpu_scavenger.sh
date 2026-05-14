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

FIG2_LIVER_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/liver"
FIG2_BRAIN_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/brain"
FIG2_GENE_IDS="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt"
FIG2_OLD_EVAL_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2/evaluation"

DATASET_ARGS=()

run_one_model () {
  MODEL_TAG="$1"
  RESULT_DIR="$2"
  MODEL_FAMILY="${3:-auto}"
  CKPT_PATH="${RESULT_DIR}/last_model.pth"
  FIG2_OUT_DIR="/hpc/group/xielab/xj58/xVerse_results/fig2_gmmvae_${MODEL_TAG}"
  FIG2_EVAL_DIR="${FIG2_OUT_DIR}/evaluation_scib_full"

  echo ">>> [${MODEL_TAG}] Extract embeddings from ${CKPT_PATH} (${MODEL_FAMILY})"
  python reproduce_manuscript/fig2_biology_signal_gmmvae_current/02_extract_gmmvae_embedding.py \
      --ckpt "${CKPT_PATH}" \
      --model-family "${MODEL_FAMILY}" \
      --gene-ids-path "${FIG2_GENE_IDS}" \
      --liver-dir "${FIG2_LIVER_DIR}" \
      --brain-dir "${FIG2_BRAIN_DIR}" \
      "${DATASET_ARGS[@]}" \
      --output-dir "${FIG2_OUT_DIR}" \
      --embedding-key xVerse_gmmvae_mixmu

  echo ">>> [${MODEL_TAG}] Evaluate FMs + GMVAE (scIB full)"
  python reproduce_manuscript/fig2_biology_signal_gmmvae_current/03_evaluate_scib_full.py \
      --liver-dir "${FIG2_LIVER_DIR}" \
      --brain-dir "${FIG2_BRAIN_DIR}" \
      --output-dir "${FIG2_EVAL_DIR}" \
      --old-eval-dir "${FIG2_OLD_EVAL_DIR}" \
      "${DATASET_ARGS[@]}" \
      --gmm-key xVerse_gmmvae_mixmu \
      --max-cells 20000 \
      --skip-official-metrics-all

  echo ">>> [${MODEL_TAG}] Plot UMAP"
  python reproduce_manuscript/fig2_biology_signal_gmmvae_current/04_plot_umap.py \
      --liver-dir "${FIG2_LIVER_DIR}" \
      --brain-dir "${FIG2_BRAIN_DIR}" \
      --output-dir "${FIG2_OUT_DIR}/umap" \
      --embedding-key xVerse_gmmvae_mixmu \
      --gene-set all \
      --max-cells 20000
}

run_one_model "base" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue0513_base" "main_energy"
# run_one_model "mfa_rank4" "/hpc/group/xielab/xj58/pretrain_model_celltype/mfa_all_tissue0513_rank4" "main_mfa"
# run_one_model "nb" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue0512" "main_energy"
# run_one_model "poisson" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue_poisson"


# python main_energy/diagnose_ckpt_val.py \
#   --ckpt /hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue2/last_model.pth \
#   --compiled-dataset-root /hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all \
#   --val-num-workers 4
