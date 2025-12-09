# Quick Start Guide

### Step 1: Install Dependencies

```bash
cd pneumonia_classification_project
pip install -r requirements.txt
```

### Step 2: Prepare Your Data

Organize your chest X-ray images:

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### Step 3: Train the Model

```bash
python scripts/train.py --data_dir ./data --batch_size 64 --num_epochs 150
```

### Step 4: Evaluate

```bash
python scripts/evaluate.py \
    --model_path ./results/checkpoints/best_model.pth \
    --data_dir ./data \
    --output_path ./results/evaluation_results.json
```

### Step 5: View Results

```bash
tensorboard --logdir ./results/logs
```

## What to Expect

**Training Time:** ~4.5 hours on 4× NVIDIA A100 GPUs  
**Expected Accuracy:** 97.23% ± 0.22%  
**Expected AUC:** 0.981  
**Calibration (ECE):** 0.016

## Key Features

- **Adaptive Augmentation:** RL agent learns optimal augmentation policies
- **Clinical Constraints:** All transformations preserve anatomical plausibility
- **Curriculum Learning:** Progressive difficulty with safety rollback
- **Uncertainty Quantification:** MC Dropout for calibrated predictions

## Configuration

Edit `config/config.yaml` to customize hyperparameters.

## Need Help?

- Check the full [README.md](README.md) for detailed documentation
- Contact: m.mahmoud@mau.edu.eg

## Reproduce Paper Results

To reproduce the exact results from the paper:

```bash
# Train with all 5 random seeds
for seed in 42 123 2025 314 555; do
    python scripts/train.py \
        --data_dir ./data \
        --seed $seed \
        --checkpoint_dir ./results/seed_$seed
done

# Evaluate each model
for seed in 42 123 2025 314 555; do
    python scripts/evaluate.py \
        --model_path ./results/seed_$seed/best_model.pth \
        --data_dir ./data \
        --compute_ci
done
```

## Monitor Training

Watch key metrics in real-time:
- Training/validation accuracy and loss
- AUC-ROC and calibration error
- PPO policy losses and entropy
- Curriculum stage progression

## Performance Tips

1. **Use GPU:** Training on CPU will be ~50× slower
2. **Batch Size:** Adjust based on GPU memory (32-128)
3. **Workers:** Set `num_workers=4` for faster data loading
4. **Mixed Precision:** Enable for faster training (requires PyTorch 1.6+)

