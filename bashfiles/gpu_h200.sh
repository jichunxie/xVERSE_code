#!/bin/bash
#SBATCH -p scavenger-h200                      
#SBATCH -A scavenger-h200                             
#SBATCH --gres=gpu:h200:1
#SBATCH -c 20                                 
#SBATCH --mem=300G                        
#SBATCH -t 23:00:00                            
#SBATCH -J h200
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt  
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt  
#SBATCH --mail-user=xj58@duke.edu                 
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --exclude=dcc-h200-gpu-05                 

# Load environment
source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code

echo ">>> Running Task: Train GMM prior model (all tissue, DDP on H200, compiled dataset mode)"
NPROC_PER_NODE=$(python - <<'PY'
import torch
print(max(1, torch.cuda.device_count()))
PY
)
echo ">>> Visible CUDA devices: ${NPROC_PER_NODE}"

DATA_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor"
COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"

echo ">>> DATA_ROOT=${DATA_ROOT}"
echo ">>> COMPILED_ROOT=${COMPILED_ROOT}"

stdbuf -oL -eL torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m main_energy.train_pantissue \
    --compiled-dataset-root "${COMPILED_ROOT}" \
    --compiled-max-cached-shards 4092 \
    --sampler-shard-reorder-window 16384 \
    --sampler-active-shards 16 \
    --result-dir "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue_h200" \
    --num-epochs 100 \
    --batch-size 2048 \
    --val-batch-size 2048 \
    --num-workers 10 \
    --val-num-workers 5 \
    --prefetch-factor 8 \
    --samples-per-id 10000 \
    --lr 3e-4 \
    --weight-decay 1e-5 \
    --prior-type gmm \
    --latent-dim 128 \
    --num-components 16 \
    --prior-cov-rank 8 \
    --expr-hidden-dim 1024 \
    --mask-hidden-dim 512 \
    --dec-hidden-dim 1024 \
    --beta-kl 0.1 \
    --recon-observed-only \
    --mask-aug-prob 1.0 \
    --mask-aug-policy xverse \
    --mask-aug-min-frac 0.1 \
    --mask-aug-max-frac 0.5 \
    --lambda-score 0 \
    --lambda-cov 0.1 \
    --score-noise-std 0.1 \
    --lambda-contrast 0.0 \
    --lambda-real-recon 0.0 \
    --contrast-temp 0.3
