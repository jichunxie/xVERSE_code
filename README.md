<p align="center">
  <img src="logo.png" width="800" alt="xVERSE Logo">
</p>

# xVERSE: Transcriptomics-Native Foundation Model

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**xVERSE** (X-Verse) is a **transcriptomics-native foundation model** designed to learn robust, batch-invariant biological representations and synthesize high-fidelity virtual cells. By coupling representation learning with probabilistic gene expression generation, xVERSE enables advanced downstream applications in single-cell and spatial transcriptomics.

## 🚀 Key Capabilities

*   **Universal Representation Learning**: Extract biological embeddings (`z_bio`) that are robust to batch effects and noise.
*   **Spatial Gene Imputation**: Inaccurately impute unmeasured genes in spatial transcriptomics data using single-cell references.
*   **Virtual Cell Synthesis**: Generate realistic, high-fidelity virtual cells to augment small datasets or serve as a data-augmentation engine.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/jichunxie/xVERSE_code.git
    cd xVERSE_code
    ```

2.  **Create a virtual environment (Recommended)**:
    ```bash
    conda create -n xverse python=3.9
    conda activate xverse
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🛠️ Usage

xVERSE provides a unified CLI `main.finetune` for all core tasks.

### Input Requirements
*   **Format**: `.h5ad` (AnnData).
*   **Annotations**:
    *   `adata.obs['tissue']`: Tissue label (must match `main/tissue_name_to_id_map.csv`).
    *   `adata.var['gene_ids']`: Ensembl IDs (e.g., `ENSG00000123456`).

### Basic Command

```bash
python -m main.finetune --input_dir <INPUT> --output_dir <OUTPUT> --base_model <MODEL_PATH> [OPTIONS]
```

### Examples

#### 1. Zero-Shot Embedding Extraction
Extract biological embeddings without fine-tuning.
```bash
python -m main.finetune \
    --input_dir ./data/liver_samples \
    --output_dir ./results/embeddings \
    --base_model ./checkpoints/xverse_pretrained.pth \
    --tissue_name liver \
    --mode 0shot \
    --task embedding
```

#### 2. Fine-Tuning & Imputation
Fine-tune on your data to generate denoised expression (`mu_bio`).
```bash
python -m main.finetune \
    --input_dir ./data/liver_samples \
    --output_dir ./results/imputation \
    --base_model ./checkpoints/xverse_pretrained.pth \
    --tissue_name liver \
    --mode ft \
    --task generation \
    --num_samples_gen 5
```

See [Examples](#examples) in the full documentation for more scenarios.

| Argument | Description | Required |
| :--- | :--- | :--- |
| `--input_dir` | Input directory or file path. | Yes |
| `--output_dir` | Output directory. | Yes |
| `--base_model` | Path to pretrained model checkpoint. | Yes |
| `--task` | `embedding` or `generation`. | Yes |
| `--mode` | `0shot` or `ft` (Fine-tune). | No (Default: 0shot) |

---

## 📂 Repository Structure

```
xVERSE_code/
├── main/                           # Core xVERSE source code
│   ├── finetune.py                 # Main CLI entry point
│   ├── utils_model.py              # Model architecture definitions
│   └── ...
├── reproduce_manuscript/           # Scripts to reproduce paper figures
│   ├── fig1_overview/              # Figure 1: Model Overview
│   ├── fig2_biology_signal/        # Figure 2: Biological Signal & Benchmarking
│   ├── fig3_check_score_for_panel/ # Figure 3: Panel Analysis
│   ├── fig4_generate_single_cell/  # Figure 4: SC Generation
│   ├── fig5_imputation_spatial/    # Figure 5: Spatial Imputation
│   ├── fig6_small_sample/          # Figure 6: Small Sample Learning
│   └── fig7_cross_modality/        # Figure 7: Cross-Modality Prediction
├── bashfiles/                      # HPC Slurm/Bash scripts
├── requirements.txt                # Python dependencies
├── LICENSE                         # GNU GPL-3.0 License
└── README.md                       # Project documentation
```

## ⚖️ License

This project is open source under the **GNU General Public License v3.0 (GPL-3.0)** - see the [LICENSE](LICENSE) file for details.

> [!NOTE]
> **Commercial Use**: This software is free for non-commercial use. For commercial use, please contact the authors to obtain a separate license:
> *   **Jichun Xie**: [jichun.xie@duke.edu](mailto:jichun.xie@duke.edu)
> *   **Xiaohui Jiang**: [x.jiang@duke.edu](mailto:x.jiang@duke.edu)
