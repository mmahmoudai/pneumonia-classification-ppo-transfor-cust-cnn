# Chest X-Ray Pneumonia Classification with RL-based Augmentation

This project implements the research framework described in "Clinically Constrained Reinforcement Learning for Optimized Adaptive Data Augmentation in Chest X-Ray Pneumonia Classification" by Muhammad Mahmoud.

## Architecture Overview

```
Input: Chest X-Ray Image (224×224×3)
    ↓
[Preprocessing & Normalization]
    ↓
[RL Agent] → Selects Augmentation Action (1 of 60)
    ↓
[Clinically Constrained Augmentation]
    ↓
[CNN Backbone] → Feature Extraction (H'×W'×256)
    ↓
[Transformer Encoder] → Global Context (N×256)
    ↓
[Classification Head] → Prediction (2 classes)
    ↓
[Reward Computation] → Feedback to RL Agent
    ↓
Output: Pneumonia/Normal + Uncertainty
```

## Key Implementation Details

### Model Architecture
- **Input:** 224×224 RGB images (grayscale replicated)
- **CNN:** 4 blocks (32→64→128→256 channels)
- **Transformer:** 2 layers, 4 heads, 256-dim embeddings
- **Output:** Binary classification (Normal/Pneumonia)

### RL Framework
- **Algorithm:** Proximal Policy Optimization (PPO)
- **State Space:** 100-dim (image stats, lung features, uncertainty, dynamics)
- **Action Space:** 60 discrete actions (5 types × 12 levels)
- **Reward:** Multi-objective (accuracy, diversity, calibration, cost, validity)

### Training Configuration
- **Epochs:** 150 (10 pre-training + 140 RL)
- **Batch Size:** 64 (classifier), 128 (PPO)
- **Learning Rates:** 2e-4 (classifier), 1e-4 (actor), 5e-4 (critic)
- **Curriculum:** 5 stages with safety rollback

### Augmentation Constraints
| Type | Range | Rationale |
|------|-------|-----------|
| Rotation | [-8°, +8°] | Patient positioning |
| Brightness | [0.9, 1.2] | Exposure variation |
| Contrast | [0.8, 1.2] | Scanner differences |
| Noise | σ ∈ [0, 0.03] | Acquisition noise |
| Blur | σ ∈ [0, 1.0] | Motion artifacts |

## Performance

### Internal Test Set (CheXpert + MIMIC-CXR)
- **Accuracy:** 97.23% (95% CI: 97.01–97.45)
- **AUC:** 0.981
- **Sensitivity:** 97.1%
- **Specificity:** 97.4%
- **ECE:** 0.016
- **Brier Score:** 0.028

### External Validation
| Dataset | Images | AUC | ECE | Accuracy |
|---------|--------|-----|-----|----------|
| ChestX-ray14 | 5,000 | 0.952 | 0.028 | 94.8% |
| VinDr-CXR | 3,000 | 0.938 | 0.034 | 92.6% |
| Kermany | 1,172 | 0.961 | 0.022 | 95.7% |

### Reproducibility
- **5 random seeds:** 97.01–97.45% (mean: 97.23% ± 0.22%)
- **Cohen's d:** 2.14 vs. baseline (large effect size)
- **Statistical significance:** p < 0.001 (McNemar's test)

## How to Use

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python scripts/train.py --data_dir ./data --batch_size 64 --num_epochs 150

# Evaluate
python scripts/evaluate.py --model_path ./results/checkpoints/best_model.pth --data_dir ./data
```

### Advanced Usage
```bash
# Custom configuration
python scripts/train.py --config config/config.yaml

# External validation
python scripts/evaluate.py --model_path best_model.pth --data_dir ./external_cohort --compute_ci

# Monitor training
tensorboard --logdir ./results/logs
```

## Usage Examples

### Training
```python
from scripts.train import RLAugmentationTrainer
from utils.dataset import create_dataloaders

# Load data
train_loader, val_loader, _ = create_dataloaders('./data', batch_size=64)

# Create trainer
trainer = RLAugmentationTrainer(config)

# Train
trainer.train(train_loader, val_loader)
```

### Inference
```python
from models.cnn_transformer import create_model
import torch

# Load model
model = create_model(num_classes=2)
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['classifier_state_dict'])

# Predict with uncertainty
mean_pred, uncertainty = model.forward_with_uncertainty(image, num_samples=10)
```

### Custom Augmentation
```python
from utils.augmentation import ClinicallyConstrainedAugmentation

aug = ClinicallyConstrainedAugmentation()
augmented_image = aug.apply_augmentation(image, action=15)
```

## Future Work

- Multi-label classification for thoracic diseases
- Extension to CT, MRI, and ultrasound modalities
- Federated learning for privacy-preserving training
- Attention visualization for explainability

## Contact

Muhammad Mahmoud  
Department of Information Systems  
Matrouh University, Egypt  
m.mahmoud@mau.edu.eg
