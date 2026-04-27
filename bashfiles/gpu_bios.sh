#!/bin/bash
#SBATCH -p biostat-gpu                     
#SBATCH -A biostat                           
#SBATCH --gres=gpu:1                        
#SBATCH -c 20                              
#SBATCH --mem=200G                        
#SBATCH -t 20:00:00                            
#SBATCH -J bios
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

echo ">>> Running Task: Train GMM-VAE (compiled dataset mode)"
NPROC_PER_NODE=$(python - <<'PY'
import torch
try:
    n = int(torch.cuda.device_count())
except Exception:
    n = 0
print(max(1, n))
PY
)
if [ -z "${NPROC_PER_NODE}" ]; then
  NPROC_PER_NODE=1
fi
echo ">>> Visible CUDA devices: ${NPROC_PER_NODE}"
echo ">>> COMPILED_ROOT=${COMPILED_ROOT}"
echo ">>> RESULT_DIR=${RESULT_DIR}"

stdbuf -oL -eL torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m main_energy.train_pantissue \
    --compiled-dataset-root "${COMPILED_ROOT}" \
    --compiled-max-cached-shards 4092 \
    --sampler-shard-reorder-window 4096 \
    --sampler-active-shards 16 \
    --result-dir "${RESULT_DIR}" \
    --num-epochs 100 \
    --val-every 5 \
    --batch-size 1024 \
    --val-batch-size 1024 \
    --num-workers 8 \
    --val-num-workers 5 \
    --prefetch-factor 8 \
    --samples-per-id 500 \
    --lr 5e-4 \
    --weight-decay 1e-5 \
    --prior-type gmm \
    --latent-dim 128 \
    --num-components 32 \
    --prior-cov-rank 4 \
    --posterior-cov-rank 4 \
    --prior-logvar-min -2 \
    --prior-logvar-max 2 \
    --expr-hidden-dim 512 \
    --mask-hidden-dim 512 \
    --dec-hidden-dim 512 \
    --beta-kl 0.01 \
    --beta-kl-start 0.01 \
    --beta-kl-end 0.01 \
    --beta-kl-warmup-epochs 0 \
    --gmm-init-after-epochs 1 \
    --gmm-stage2-epochs 0 \
    --gmm-post-init-kl-warmup-epochs 0 \
    --gmm-init-max-samples 200000 \
    --gmm-init-max-batches 300 \
    --gmm-init-iters 30 \
    --recon-observed-only \
    --mask-aug-prob 1.0 \
    --mask-aug-policy simple \
    --mask-aug-min-frac 0.05 \
    --mask-aug-max-frac 0.25 \
    --lambda-score 0 \
    --lambda-cov 0 \
    --lambda-resp-balance 0 \
    --lambda-resp-balance-warmup-epochs 0 \
    --lambda-resp-confidence 0 \
    --lambda-resp-confidence-warmup-epochs 0 \
    --lambda-resp-anchor 0.0 \
    --lambda-prior-mu-l2 0 \
    --lambda-prior-factor-l2 0 \
    --lambda-prior-pi-balance 0 \
    --lambda-prior-logvar-l2 0 \
    --prior-logvar-target -2 \
    --lambda-celltype-cls 1 \
    --resp-temperature 1 \
    --resp-temperature-start 1 \
    --resp-temperature-warmup-epochs 0 \
    --score-noise-std 0.1 \
    --lambda-contrast 1 \
    --lambda-real-recon 0.0 \
    --contrast-temp 0.1


# python main_energy/diagnose_ckpt_val.py \
#   --ckpt /hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue2/last_model.pth \
#   --compiled-dataset-root /hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all \
#   --val-num-workers 4
