## Overview

This model is a **Mask-FiLM GMM-VAE** designed for sparse single-cell count data. Its high-level architecture is: mask-aware encoder + Gaussian mixture latent prior + count-distribution decoder. The inputs are `x_count` (gene counts) and `x_mask` (observation mask), where missing entries are modeled explicitly through masking instead of simple imputation.

The encoder uses a dual-branch design. The expression branch encodes `x_expr * x_mask`, while the mask branch encodes `x_mask` itself; FiLM modulation (`gamma`, `beta`) then conditions expression features with mask-derived context to produce posterior parameters. The core motivation is to distinguish biological low expression from technical missingness, improving robustness under high sparsity.

The latent prior is a learnable GMM. Each component covariance is parameterized as `diag(exp(logvar_k)) + U_k U_k^T`, i.e., a diagonal term plus a low-rank correction, balancing expressiveness and computation. The model learns mixture weights, component means, and covariance parameters, and supports top-k responsibility approximation, log-variance clamping, and prior regularization to mitigate component collapse and numerical instability.

For the posterior, when `prior_type=gmm`, the model uses a mixture posterior: it predicts `q(c|x)`, then predicts `q(z|x,c)` per component and samples `z_comp`. During training, component selection is soft via Gumbel-Softmax; during inference, hard assignment is used. This aligns the posterior with a multi-modal prior and better captures heterogeneous cellular states.

The decoder maps latent `z` to gene logits, applies softmax to obtain gene composition probabilities, and predicts library size separately; their product gives the Poisson rate for count reconstruction. The loss is primarily reconstruction + KL. In GMM mode, KL is decomposed as `KL(q(c|x)||p(c)) + E_q[KL(q(z|x,c)||p(z|c))]`. The code also includes optional terms such as score matching, responsibility balance/confidence constraints, prior parameter regularization, and auxiliary cell-type classification to improve stability and interpretability across training stages.

## Evaluating the Latent Space

To evaluate the learned latent space, we will use a multi-perspective protocol covering topology, geometry, biological signal preservation, and batch-effect removal.

Topology evaluates whether global manifold structure is preserved after embedding. We will report:
- Betti number
- Betti curve

Geometry evaluates whether pairwise and neighborhood relationships are preserved. We will report:
- Distance preservation
- Trustworthiness

Biological label conservation evaluates whether biologically meaningful labels remain separable and recoverable in latent space. We will report:
- NMI
- ARI
- cLISI
- Probing classifiers

Batch correction evaluates how well non-biological batch variation is removed while retaining biological structure. We will report:
- kBET
- PCR
- iLISI
- Graph connectivity

Together, these metrics provide a balanced view: preserving intrinsic structure and biological semantics while reducing technical confounding.

## Evaluating Cell Generation Capability

To assess how well the model generates realistic and useful synthetic cells, we will evaluate distributional fidelity, diversity, biological consistency, and practical utility.

Distributional fidelity (real vs. generated)
- MMD (Maximum Mean Discrepancy) in PCA/latent space
- Energy distance or Wasserstein distance on expression profiles
- Gene-wise distribution match (mean/variance and zero-rate gap)
- Spearman rank correlation of gene-level mean expression (real vs. generated)
- Spearman rank correlation of highly variable gene (HVG) rankings
- Rank-biased overlap (RBO) for top-k marker/HVG gene lists

Discriminative classifier
- More complex

Diversity and mode coverage
- Coverage of real-cell neighborhoods by generated cells (kNN coverage)
- Cluster-level proportion match (cell-type frequency error)
- Effective number of occupied clusters/components (mode collapse check)

Biological consistency
- Marker gene recovery score per cell type (known markers enriched in the right populations)
- Cell-type annotation agreement (transfer labels from a reference classifier)
- Pathway/activity profile similarity between real and generated cohorts
- Gene co-expression preservation (correlation matrix similarity, e.g., Pearson/Spearman)
- Co-expression module conservation (overlap of modules and intra-module connectivity)


A practical reporting strategy is to include at least one metric from each block above, so results reflect not only sample realism but also biological validity and downstream usefulness.

## Downstream Task 1: Predicting TCR/BCR/ADT from RNA Latent Features

Goal
- Evaluate whether the learned representation improves prediction from RNA to immune-relevant modalities: TCR, BCR, and ADT.

Task setup
- Use your paired dataset where RNA is matched with TCR/BCR/ADT at the cell level.
- Train modality-specific predictors on top of latent `z`:
- RNA -> TCR prediction
- RNA -> BCR prediction
- RNA -> ADT prediction
- Compare against predictors trained on raw RNA and baseline embeddings (e.g., PCA/standard VAE).
- Evaluate both random split and donor/batch-held-out split to measure generalization.

## Downstream Task 2: Improving Perturbation Prediction

Goal
- Test whether the model better predicts cellular response to genetic or chemical perturbations.

Task setup
- Train on control + perturbation data with held-out perturbations/cell contexts.
- Predict post-perturbation expression (or latent shift) from pre-perturbation state and perturbation condition.

Key metrics
- Gene-level correlation between predicted and observed perturbed profiles
- Differential expression recovery (top DE overlap, effect-direction accuracy)
- Pathway-level response consistency (enrichment score correlation)
- Perturbation retrieval/identification accuracy in latent space

Additional analysis
- OOD generalization to unseen perturbations
- Performance on rare cell states and strong perturbation regimes
- Calibration of uncertainty for high-impact response genes

## Downstream Task 3: Improving Small-Sample Analysis

Goal
- Evaluate whether the model improves analysis quality under low-data regimes.

Task setup
- Simulate limited-sample settings by subsampling cells per dataset/cell type.
- Use generated or latent-augmented data for training and compare against non-augmented baselines.

Key metrics
- Classification/clustering performance gain at fixed sample budgets
- Stability across random subsamples (variance of metrics across seeds)
- Rare-cell detection performance (recall/F1/AUPRC)
- Differential expression robustness (reproducibility across subsamples)

## Downstream Task 3: Bulk deconvolution (Rare cell type)

## Downstream Task 4: Spatial Imputation
cms4