# EuroSAT Land Cover Classification with CLIP

This project explores land cover classification using the EuroSAT satellite imagery dataset, leveraging the power of vision-language foundation models—specifically, CLIP. The goal is to evaluate and compare zero-shot and few-shot classification techniques using linear probing, MLP classifiers, logistic regression, and prompt learning via CoOp.

## Project Overview

We focus on adapting CLIP (Contrastive Language–Image Pretraining) to satellite imagery classification through multiple learning setups:

- **Zero-shot learning** using text-image similarity with handcrafted prompts.
- **Few-shot learning** via:
  - Linear probing
  - MLP probing
  - Logistic regression
- **Prompt learning** using CoOp (Context Optimization), where prompts are learned in a class-specific manner.

This approach enables effective classification without fully retraining the CLIP model, which is valuable in domains like remote sensing where labeled data is limited.

## Dataset

The [EuroSAT dataset](https://github.com/phelber/eurosat) consists of 27,000+ RGB satellite images of 64x64 pixels, each belonging to one of the following land cover classes:

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

This dataset serves as a benchmark to test the generalization capabilities of CLIP in a remote sensing context.

## Relevant Code Files

### `customCLIP.py`

This module defines the `customCLIP` class, which wraps the HuggingFace CLIP model and provides utilities for:

- Initializing the model with configurable options for prompt style, class labels, and image augmentation.
- Zero-shot classification using CLIP’s built-in similarity scoring between image features and text prompts.
- Few-shot training and evaluation using different classifiers:
  - Linear probe
  - MLP probe
  - Logistic regression
- Prompt learning through CoOp, using learnable context vectors optimized via backpropagation.
- Evaluation functions to compute per-class and full-dataset accuracy.
- Model saving and loading for reproducibility and testing.

The module also includes internal utilities for:
- Feature extraction from images
- Input preprocessing with or without prompt text
- Hue and contrast-based data augmentation

All classifier training is done with CLIP’s image encoder frozen.

### `customCLIP_demo.ipynb`

This Jupyter notebook demonstrates the usage of `customCLIP.py` and includes examples for:

- Training classifiers in few-shot settings
- Testing from pre-trained models
- Zero-shot evaluation
- Single-class and multi-class performance analysis
- Comparative plots between methods

Each section is annotated and parameterized for ease of use and reproducibility.

## Environment Setup

Dependencies are listed in `environment.yml`. Use the following commands to create and activate the environment:

```bash
conda env create -f environment.yml
conda activate land-classification
```

## Implementation Notes

- CLIP models are loaded using the HuggingFace Transformers library, specifically `openai/clip-vit-base-patch32`.
- During few-shot training, the CLIP backbone remains frozen. Only lightweight classifiers or prompt embeddings are trained, ensuring efficiency and preserving generalization.
- The CoOp method implements class-specific prompt learning by optimizing learnable context vectors that are prepended to each class label.
- Zero-shot classification performance is influenced significantly by the phrasing and formatting of text prompts, making prompt design an important factor in evaluation.

## References

This project is based on the following foundational works:

- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [CoOp: Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134)
- [ECO: Ensembling Context Optimization for Vision-Language Models](https://arxiv.org/abs/2305.13827)
