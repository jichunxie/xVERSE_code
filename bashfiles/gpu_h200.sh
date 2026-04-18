#!/bin/bash
#SBATCH -p scavenger-h200                      
#SBATCH -A scavenger-h200                             
#SBATCH --gres=gpu:h200:2
#SBATCH -c 20                                 
#SBATCH --mem=200G                        
#SBATCH -t 12:00:00                            
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

echo ">>> Running Task: Train GMM-VAE (kidney only)"
stdbuf -oL -eL python -m main_energy.train_pantissue \
    --data-root "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor" \
    --summary-csv "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/pantissue_full_updated.csv" \
    --cell-type-csv "/hpc/group/xielab/xj58/sparest_code/standard_type/cellxgene_cell_type_mapped.csv" \
    --gene-ids-path "/hpc/group/xielab/xj58/xVerseAtlas/npz_tissue_dataset_donor/ensg_keys_high_quality.txt" \
    --result-dir "/hpc/group/xielab/xj58/pretrain_model_celltype/gmmvae_kidney_h200" \
    --use-tissue "kidney" \
    --num-epochs 50 \
    --batch-size 512 \
    --val-batch-size 512 \
    --num-workers 20 \
    --samples-per-id 1000 \
    --lr 1e-4 \
    --weight-decay 1e-5 \
    --latent-dim 128 \
    --num-components 16 \
    --expr-hidden-dim 1024 \
    --mask-hidden-dim 512 \
    --dec-hidden-dim 1024 \
    --beta-kl 1.0 \
    --recon-observed-only \
    --mask-aug-prob 1.0 \
    --mask-aug-policy xverse \
    --mask-aug-min-frac 0.1 \
    --mask-aug-max-frac 0.5 \
    --lambda-score 0.05 \
    --score-noise-std 0.1
