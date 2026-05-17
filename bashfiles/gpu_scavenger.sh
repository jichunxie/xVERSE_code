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
  EMBEDDING_MODE="${4:-mixmu}"
  CKPT_NAME="${5:-best_contrast_model.pth}"
  shift 5 || true
  EXTRACT_EXTRA_ARGS=("$@")
  EMBEDDING_KEY="xVerse_gmmvae_${EMBEDDING_MODE}"
  CKPT_PATH="${RESULT_DIR}/${CKPT_NAME}"
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
      "${EXTRACT_EXTRA_ARGS[@]}" \
      --output-dir "${FIG2_OUT_DIR}" \
      --embedding-key "${EMBEDDING_KEY}" \
      --embedding-mode "${EMBEDDING_MODE}"

  echo ">>> [${MODEL_TAG}] Evaluate FMs + GMVAE (scIB full)"
  python reproduce_manuscript/fig2_biology_signal_gmmvae_current/03_evaluate_scib_full.py \
      --liver-dir "${FIG2_LIVER_DIR}" \
      --brain-dir "${FIG2_BRAIN_DIR}" \
      --output-dir "${FIG2_EVAL_DIR}" \
      --old-eval-dir "${FIG2_OLD_EVAL_DIR}" \
      "${DATASET_ARGS[@]}" \
      --gmm-key "${EMBEDDING_KEY}" \
      --max-cells 20000 \
      --skip-official-metrics-all

  echo ">>> [${MODEL_TAG}] Plot UMAP"
  python reproduce_manuscript/fig2_biology_signal_gmmvae_current/04_plot_umap.py \
      --liver-dir "${FIG2_LIVER_DIR}" \
      --brain-dir "${FIG2_BRAIN_DIR}" \
      --output-dir "${FIG2_OUT_DIR}/umap" \
      --embedding-key "${EMBEDDING_KEY}" \
      --gene-set all \
      --max-cells 20000
}

run_one_model "vae" "/hpc/group/xielab/xj58/pretrain_model_celltype/vae_all_tissue0517_con3_nobatch_kl1e-3" "main_mfa" "mu_base" "last_model.pth" --no-prior-viz
# run_one_model "mfa" "/hpc/group/xielab/xj58/pretrain_model_celltype/mfa_all_tissue0517_con2_16x4_nobatch_nobalance_randomprior5_trunk" "main_mfa" "encoder_hidden" "last_model.pth"
# run_one_model "gene_theta_encoder_0515" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue0515_2" "main_energy" "encoder_hidden"
# run_one_model "mfa_rank4" "/hpc/group/xielab/xj58/pretrain_model_celltype/mfa_all_tissue0513_rank4" "main_mfa"
# run_one_model "nb" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue0512" "main_energy"
# run_one_model "poisson" "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue_poisson"


# python main_energy/diagnose_ckpt_val.py \
#   --ckpt /hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue2/last_model.pth \
#   --compiled-dataset-root /hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all \
#   --val-num-workers 4
