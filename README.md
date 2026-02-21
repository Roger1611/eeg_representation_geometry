# EEG Motor Imagery Representation Learning with ICRR

## Overview

This repository presents a structured empirical study on EEG-based Motor Imagery (MI) classification, with a focus on representation geometry and supervised embedding regularization.

The project progresses through three experimental stages:

1. Classical Machine Learning Baselines (CSP + Linear Models)
2. Deep Learning Benchmarking (EEGNet, ShallowConvNet, DeepConvNet)
3. Intra-Class Representation Regularization (ICRR) Ablation Study

The primary contribution is:

ICRR - Intra-Class Representation Regularization

A supervised embedding constraint that promotes intra-class compactness in the latent feature space without modifying model architecture or inference cost.

---

## Datasets

### BNCI 2014-001 (Primary Dataset)

All deep learning benchmarking and representation experiments are conducted on the BNCI 2014 Motor Imagery dataset.

- 4-class motor imagery task
- 25 EEG channels
- Preprocessed input shape: (640, 25, 561)
- Classes: Left Hand, Right Hand, Both Feet, Tongue

### PhysioNet EEGBCI (Exploratory)

Early prototyping and classical baselines were conducted using PhysioNet EEGBCI.
These experiments are retained for completeness but are not part of the final representation study.

---

## Repository Structure

```
datasets/
├── bnci_dataset/
│   ├── raw/
│   └── processed/
│       └── preprocessed_BNCI.npz
├── physionet_dataset/
│   ├── raw/
│   ├── intermediate/
│   └── processed/

experiments/
├── stage1_ml_baseline/
├── stage2_dl_baseline/
└── stage3_dl_icrr/

notebooks/
├── stage1_ml_baseline/
├── stage2_dl_baseline/
└── stage3_dl_icrr/

results/
├── figures/
└── tables/

src/
├── classical_models.py
├── deep_models.py
├── preprocessing.py
├── data_loading.py
└── evaluation.py
```
---

## Experimental Pipeline

### Stage 1 — Classical Baselines

- Common Spatial Patterns (CSP) feature extraction
- LDA, SVM (RBF), Random Forest, MLP
- Stratified 5-fold cross-validation
- Metrics: Accuracy and F1-score

Purpose: establish non-deep baseline performance.

---

### Stage 2 — Deep Learning Benchmark

Architectures evaluated:

- EEGNet
- ShallowConvNet
- DeepConvNet

Training setup:

- Cross-Entropy loss
- Stratified split
- Adam optimizer
- Standard EEG normalization

Outputs:

- Fold-level checkpoints
- Benchmark summaries
- Confusion matrices

---

### Stage 3 — Intra-Class Representation Regularization (ICRR)

ICRR introduces a supervised embedding constraint to DeepConvNet:

- Pulls same-class embeddings toward their class centroid
- Encourages compact intra-class representation structure
- Leaves architecture unchanged
- Adds no additional inference overhead

### Loss Formulation

$$
\mathcal{L}_{\text{total}} 
= 
\mathcal{L}_{\text{CE}} 
+ 
\lambda \, \mathcal{L}_{\text{ICRR}}
$$

Where:

- L_CE = Cross-Entropy loss
- L_ICRR = Intra-class embedding compactness term

---

## Key Results

### Validation Accuracy

DeepConvNet (CE): ~0.70  
DeepConvNet (CE + ICRR): comparable performance

### Embedding Stability Score (ESS)

| Method      | ESS     |
|-------------|---------|
| CE          | 39.30   |
| CE + ICRR   | 32.25   |

Lower ESS indicates tighter intra-class clustering.

### UMAP Visualization

- Baseline (CE) shows dispersed class manifolds.
- CE + ICRR produces visibly more compact clusters.

All figures are available in:

results/figures/

---

## Reproducibility

Install dependencies:

pip install -r requirements.txt

Run experiments in order:

1. notebooks/stage1_ml_baseline/
2. notebooks/stage2_dl_baseline/
3. notebooks/stage3_dl_icrr/

Final ablation outputs are stored in:

experiments/stage3_dl_icrr/

---

## Methodological Scope

This work does not claim:

- Intent inference
- Cognitive state modeling
- Temporal mental state decoding

ICRR is strictly a supervised representation-level regularization technique that improves intra-class compactness in EEG embedding space.

---

## Research Motivation

While many EEG classification studies focus solely on accuracy, representation geometry plays a critical role in robustness and stability.

This study investigates:

- The effect of supervised embedding constraints on latent geometry
- Intra-class compactness under regularization
- Manifold structure changes visualized via UMAP
- Stability analysis using ESS

---

## Future Work

- Cross-subject generalization
- Robustness to signal noise
- Transformer-based EEG encoders
- Comparison with metric learning frameworks
- Representation-level interpretability analysis

---

## License

Intended for academic and research use.