# Chest X-Ray Pneumonia Classification with Clinically Constrained RL-based Augmentation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

This project implements a **clinically constrained reinforcement learning framework** for adaptive data augmentation in chest X-ray pneumonia classification, based on the research paper "Clinically Constrained Reinforcement Learning for Optimized Adaptive Data Augmentation in Chest X-Ray Pneumonia Classification" by Muhammad Mahmoud.

## Project Overview

The framework combines:
- **CNN-Transformer Hybrid Architecture** for robust feature extraction and global context modeling
- **PPO-based RL Agent** for adaptive augmentation policy learning
- **Clinically Constrained Action Space** preserving anatomical plausibility
- **Curriculum Learning** with safety rollback mechanisms
- **Multi-Objective Reward Function** balancing accuracy, diversity, calibration, and clinical validity

### Key Features

- **97.23% accuracy** on internal test set (191K training images)
- **Excellent calibration** (ECE = 0.016)
- **Robust generalization** across external cohorts (AUC = 0.950 average)
- **Balanced performance** (Sensitivity: 97.1%, Specificity: 97.4%)
- **Clinically validated** augmentation constraints from radiological literature

## Project Structure

```
pneumonia_classification_project/
│
├── data/                          # Dataset directory
│   ├── train/                     # Training data
│   ├── val/                       # Validation data
│   └── test/                      # Test data
│
├── models/                        # Model architectures
│   ├── cnn_transformer.py         # CNN-Transformer hybrid classifier
│   └── ppo_agent.py               # PPO-based RL agent
│
├── utils/                         # Utility modules
│   ├── dataset.py                 # Dataset loading and state extraction
│   ├── augmentation.py            # Clinically constrained augmentation
│   └── metrics.py                 # Evaluation metrics and calibration
│
├── scripts/                       # Training and evaluation scripts
│   ├── train.py                   # Main training script
│   └── evaluate.py                # Model evaluation script
│
├── config/                        # Configuration files
│   └── config.yaml                # Hyperparameters and settings
│
├── results/                       # Training outputs
│   ├── checkpoints/               # Model checkpoints
│   ├── logs/                      # TensorBoard logs
│   ├── figures/                   # Generated figures
│   └── evaluation/                # Evaluation metrics JSON files
│
└── README.md                      # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd pneumonia_classification_project
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare your dataset:**

Organize your chest X-ray dataset in the following structure:
```
data/
├── train/
│   ├── NORMAL/
│   │   ├── image1.jpg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── image1.jpg
│       └── ...
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

**Supported Datasets:**
- CheXpert (Stanford)
- MIMIC-CXR (MIT/BIDMC)
- ChestX-ray14 (NIH)
- VinDr-CXR (Vietnam)
- Kermany Pediatric Dataset (Guangzhou)

## Training

### Basic Training

```bash
python scripts/train.py \
    --data_dir ./data \
    --batch_size 64 \
    --num_epochs 150 \
    --lr 0.0002 \
    --checkpoint_dir ./results/checkpoints \
    --log_dir ./results/logs
```

### Training Phases

1. **Pre-training (Epochs 1-10):** Classifier trained without RL augmentation
2. **RL Augmentation (Epochs 11-150):** PPO agent learns adaptive augmentation policies
3. **Curriculum Progression:** Difficulty increases every 20 epochs with safety rollback

### Monitoring Training

View training progress with TensorBoard:
```bash
tensorboard --logdir ./results/logs
```

Metrics tracked:
- Training/validation loss and accuracy
- AUC-ROC and calibration error (ECE)
- PPO policy/value losses and entropy
- Curriculum stage progression

## Evaluation

### Basic Evaluation

```bash
python scripts/evaluate.py \
    --model_path ./results/checkpoints/best_model.pth \
    --data_dir ./data \
    --batch_size 64 \
    --output_path ./results/evaluation_results.json
```

### Evaluation with Uncertainty Quantification

```bash
python scripts/evaluate.py \
    --model_path ./results/checkpoints/best_model.pth \
    --data_dir ./data \
    --mc_samples 10 \
    --compute_ci \
    --output_path ./results/evaluation_results.json
```

### Evaluation Metrics

The evaluation script computes:
- **Classification metrics:** Accuracy, Precision, Recall, F1-Score
- **Clinical metrics:** Sensitivity, Specificity, PPV, NPV
- **Calibration metrics:** ECE, Brier Score
- **Uncertainty quantification:** MC Dropout with confidence intervals
- **AUC-ROC:** Area under the receiver operating characteristic curve

## Model Architecture

### CNN-Transformer Classifier

**Parameters:** 340,899

**Architecture:**
- **CNN Backbone:** 4-layer convolutional feature extractor (32→64→128→256 channels)
- **Transformer Encoder:** 2 layers, 4 attention heads, 256-dim embeddings
- **Classification Head:** LayerNorm → Dropout → Linear(256→128) → GELU → Linear(128→2)
- **Uncertainty:** Monte Carlo Dropout for calibrated predictions

### PPO Agent

**Parameters:** 173,117

**Architecture:**
- **State Space:** 100-dimensional vector (image stats, lung features, uncertainty, dynamics)
- **Action Space:** 60 discrete actions (5 augmentation types × 12 intensity levels)
- **Network:** 3-layer MLP (100→256→256→256) with separate policy/value heads

## Augmentation Constraints

All augmentation parameters are derived from radiological literature:

| Transformation | Range | Clinical Rationale |
|---------------|-------|-------------------|
| **Rotation** | [-8°, +8°] | Mimics patient positioning variability |
| **Brightness** | [0.9, 1.2] | Preserves lung field visibility |
| **Contrast** | [0.8, 1.2] | Maintains pathological opacity visibility |
| **Gaussian Noise** | σ ∈ [0, 0.03] | Realistic acquisition noise |
| **Gaussian Blur** | σ ∈ [0, 1.0] | Minor motion artifacts |

**Note:** Horizontal flipping is explicitly excluded as it inverts anatomical laterality.

## Expected Results

### Internal Test Set (CheXpert + MIMIC-CXR)

- **Accuracy:** 97.23% (95% CI: 97.01–97.45)
- **AUC:** 0.981
- **Sensitivity:** 97.1%
- **Specificity:** 97.4%
- **ECE:** 0.016

### External Validation

| Dataset | AUC | ECE | Accuracy |
|---------|-----|-----|----------|
| ChestX-ray14 (NIH) | 0.952 | 0.028 | 94.8% |
| VinDr-CXR (Vietnam) | 0.938 | 0.034 | 92.6% |
| Kermany Pediatric | 0.961 | 0.022 | 95.7% |

## Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  num_classes: 2
  img_size: 224
  base_channels: 32
  transformer_dim: 256
  nhead: 4
  num_transformer_layers: 2
  dropout: 0.3

training:
  batch_size: 64
  num_epochs: 150
  classifier_lr: 0.0002
  rl_start_epoch: 10

reward:
  alpha: 1.0    # Accuracy improvement
  beta: 0.3     # Augmentation diversity
  delta: 0.5    # Calibration improvement
  gamma: 0.1    # Computational cost
  epsilon: 2.0  # Clinical implausibility penalty
```

## Advanced Usage

### Custom Dataset

Create a CSV file with columns: `image_path`, `label`

```python
from utils.dataset import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    data_dir='./data',
    batch_size=64,
    train_csv='./data/train.csv',
    val_csv='./data/val.csv',
    test_csv='./data/test.csv'
)
```

### Fine-tuning on External Cohort

```bash
python scripts/train.py \
    --data_dir ./external_data \
    --checkpoint_path ./results/checkpoints/best_model.pth \
    --num_epochs 50 \
    --lr 0.00005
```

### Reproducibility

The framework uses fixed random seeds for reproducibility:
```python
seeds = [42, 123, 2025, 314, 555]
```

Results across 5 runs: **97.23% ± 0.22%** (range: 97.01–97.45%)

## Citation

If you use this code in your research, please cite:

### Software Citation (DOI)

```bibtex
@software{mahmoud2025pneumonia_rl,
  author       = {Mahmoud, Muhammad},
  title        = {{Clinically Constrained Reinforcement Learning for Optimized Adaptive Data Augmentation in Chest X-Ray Pneumonia Classification}},
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

### Research Paper Citation

```bibtex
@article{mahmoud2025clinically,
  title={Clinically Constrained Reinforcement Learning for Optimized Adaptive Data Augmentation in Chest X-Ray Pneumonia Classification},
  author={Mahmoud, Muhammad},
  journal={Under Review},
  year={2025}
}
```

> **Note:** After the first release is published on Zenodo, the DOI badge and citation will be automatically updated with the actual DOI number.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

**Muhammad Mahmoud**  
Department of Information Systems  
Faculty of Computers and Artificial Intelligence  
Matrouh University, Egypt  
Email: m.mahmoud@mau.edu.eg

## Acknowledgments

This study utilized publicly available datasets:
- CheXpert from Stanford AIMI
- MIMIC-CXR from PhysioNet
- ChestX-ray14 from NIH Clinical Center
- VinDr-CXR from PhysioNet
- Chest X-Ray Images (Pneumonia) from Kaggle

## Documentation

For detailed methodology and results, refer to:
- **Training Logs:** `results/logs/` (TensorBoard)
- **Evaluation Results:** `results/evaluation/` (JSON files)
- **Generated Figures:** `results/figures/`

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/train.py --batch_size 32
```

**2. Dataset Not Found**
```bash
# Verify directory structure
ls data/train/NORMAL
ls data/train/PNEUMONIA
```

**3. Slow Training**
```bash
# Reduce number of workers or use CPU
python scripts/train.py --num_workers 2
```

## Future Work

- Multi-label thoracic disease classification
- Federated learning for privacy-preserving training
- Explainability with attention visualization
- Extension to CT, MRI, and ultrasound modalities
- Few-shot learning for rare diseases
- Prospective clinical validation trials
