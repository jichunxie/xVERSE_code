#!/bin/bash
#SBATCH -p biostat-gpu
#SBATCH -A biostat
#SBATCH --gres=gpu:1
#SBATCH -c 10
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -J bios_mfa
#SBATCH --output=/hpc/group/xielab/xj58/sbatch_output/%x_output_%j.txt
#SBATCH --error=/hpc/group/xielab/xj58/sbatch_output/%x_error_%j.txt
#SBATCH --mail-user=xj58@duke.edu
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code
export PYTHONUNBUFFERED=1

COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"
CELLTYPE_CSV="/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv"
RESULT_DIR="/hpc/group/xielab/xj58/pretrain_model_celltype/mfa_all_tissue0513_rank4"

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

echo ">>> Running pretraining"
echo ">>> NPROC_PER_NODE=${NPROC_PER_NODE}"
echo ">>> COMPILED_ROOT=${COMPILED_ROOT}"
echo ">>> RESULT_DIR=${RESULT_DIR}"

if [ "${NPROC_PER_NODE}" -le 1 ]; then
  echo ">>> Single GPU mode: use python -m (no torchrun)"
  RUN_CMD=(python -u -m main_mfa.train_pantissue)
else
  echo ">>> Multi-GPU mode: use torchrun"
  RUN_CMD=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m main_mfa.train_pantissue)
fi

stdbuf -oL -eL "${RUN_CMD[@]}" \
  --compiled-dataset-root "${COMPILED_ROOT}" \
  --compiled-max-cached-shards 4092 \
  --sampler-shard-reorder-window 4096 \
  --cell-type-csv "${CELLTYPE_CSV}" \
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
  --recon-loss nb \
  --latent-dim 128 \
  --num-components 64 \
  --prior-cov-rank 4 \
  --batch-emb-dim 32 \
  --batch-cond-drop-prob 0.0 \
  --lambda-batchless-recon 0.1 \
  --prior-logvar-max 1 \
  --expr-hidden-dim 512 \
  --mask-hidden-dim 512 \
  --dec-hidden-dim 512 \
  --beta-kl 0.001 \
  --recon-observed-only \
  --mask-aug-prob 1.0 \
  --mask-aug-policy simple \
  --mask-aug-min-frac 0.1 \
  --mask-aug-max-frac 0.5 \
  --lambda-celltype-cls 1 \
  --lambda-contrast 1 \
  --lambda-real-recon 0.1 \
  --lambda-prior-pi-balance 0 \
  --lambda-prior-mu-spread 0 \
  --prior-mu-spread-tau 1.0 \
  --lambda-post-c-balance 0 \
  --recon-gene-weight-mode inv_log1p_mean_ema \
  --recon-gene-weight-alpha 0.8 \
  --recon-cell-weight-mode batch_kmeans \
  --recon-cell-weight-alpha 0.5 \
  --recon-cell-weight-clusters 32 \
  --recon-cell-weight-kmeans-iters 5 \
  --contrast-temp 0.1
