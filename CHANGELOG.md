# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-14

### Added

- **CNN-Transformer Hybrid Classifier**
  - 4-layer CNN backbone (32→64→128→256 channels)
  - 2-layer Transformer encoder with 4 attention heads
  - Monte Carlo Dropout for uncertainty quantification
  - Total parameters: 340,899

- **PPO-based Reinforcement Learning Agent**
  - 100-dimensional state space (image statistics, lung features, uncertainty, dynamics)
  - 60 discrete actions (5 augmentation types × 12 intensity levels)
  - 3-layer MLP architecture with separate policy/value heads
  - Total parameters: 173,117

- **Clinically Constrained Augmentation Module**
  - Rotation: [-8°, +8°] (mimics patient positioning variability)
  - Brightness: [0.9, 1.2] (preserves lung field visibility)
  - Contrast: [0.8, 1.2] (maintains pathological opacity visibility)
  - Gaussian Noise: σ ∈ [0, 0.03] (realistic acquisition noise)
  - Gaussian Blur: σ ∈ [0, 1.0] (minor motion artifacts)

- **Training Framework**
  - Pre-training phase (epochs 1-10) without RL
  - RL augmentation phase (epochs 11-150) with adaptive policy learning
  - Curriculum learning with 5 stages and safety rollback mechanisms
  - Multi-objective reward function (accuracy, diversity, calibration, clinical validity)

- **Evaluation Metrics**
  - Classification metrics: Accuracy, Precision, Recall, F1-Score
  - Clinical metrics: Sensitivity, Specificity, PPV, NPV
  - Calibration metrics: ECE, Brier Score
  - Uncertainty quantification with confidence intervals

- **Documentation**
  - Comprehensive README with usage instructions
  - Project summary and quick start guide
  - LaTeX documentation for methodology

- **Configuration**
  - YAML-based configuration file
  - Reproducibility with fixed random seeds
  - Customizable hyperparameters

### Performance Results

- **Internal Test Set (CheXpert + MIMIC-CXR)**
  - Accuracy: 97.23% (95% CI: 97.01–97.45%)
  - AUC: 0.981
  - Sensitivity: 97.1%
  - Specificity: 97.4%
  - ECE: 0.016

- **External Validation**
  - ChestX-ray14 (NIH): AUC 0.952, Accuracy 94.8%
  - VinDr-CXR (Vietnam): AUC 0.938, Accuracy 92.6%
  - Kermany Pediatric: AUC 0.961, Accuracy 95.7%

[1.0.0]: https://github.com/mmahmoudai/pneumonia-classification-ppo-transfor-cust-cnn/releases/tag/v1.0.0
