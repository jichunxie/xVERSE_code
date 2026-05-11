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

source ~/.bashrc
conda activate SpaRest

cd /hpc/group/xielab/xj58/xVERSE_code
export PYTHONUNBUFFERED=1

COMPILED_ROOT="/hpc/group/xielab/xj58/xVerseAtlas/compiled_train_v1_all"
CELLTYPE_CSV="/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv"
RESULT_DIR="/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_all_tissue0511_poisson"

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
  RUN_CMD=(python -u -m main_energy.train_pantissue)
else
  echo ">>> Multi-GPU mode: use torchrun"
  RUN_CMD=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" -m main_energy.train_pantissue)
fi

stdbuf -oL -eL "${RUN_CMD[@]}" \
  --compiled-dataset-root "${COMPILED_ROOT}" \
  --compiled-max-cached-shards 16 \
  --sampler-shard-reorder-window 8192 \
  --cell-type-csv "${CELLTYPE_CSV}" \
  --result-dir "${RESULT_DIR}" \
  --num-epochs 100 \
  --val-every 5 \
  --batch-size 1024 \
  --val-batch-size 1024 \
  --num-workers 4 \
  --val-num-workers 2 \
  --prefetch-factor 2 \
  --no-persistent-workers \
  --samples-per-id 500 \
  --lr 5e-4 \
  --weight-decay 1e-5 \
  --prior-type gmm \
  --recon-loss poisson \
  --latent-dim 128 \
  --num-components 32 \
  --prior-cov-rank 8 \
  --posterior-cov-rank 8 \
  --prior-logvar-max 2 \
  --expr-hidden-dim 512 \
  --mask-hidden-dim 512 \
  --dec-hidden-dim 512 \
  --beta-kl 0.001 \
  --recon-observed-only \
  --mask-aug-prob 1.0 \
  --mask-aug-policy simple \
  --mask-aug-min-frac 0.1 \
  --mask-aug-max-frac 0.5 \
  --lambda-celltype-cls 0 \
  --lambda-contrast 1 \
  --contrast-temp 0.1
