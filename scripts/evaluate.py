import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import argparse
import json

from models.cnn_transformer import create_model
from utils.dataset import create_dataloaders
from utils.metrics import compute_metrics, compute_calibration_error, print_metrics, compute_brier_score


class ModelEvaluator:
    def __init__(self, model_path, config, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = create_model(
            num_classes=config['num_classes'],
            img_size=config['img_size'],
            base_channels=config['base_channels'],
            transformer_dim=config['transformer_dim'],
            nhead=config['nhead'],
            num_transformer_layers=config['num_transformer_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['classifier_state_dict'])
        self.model.eval()
        
        print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    
    def evaluate(self, test_loader, num_mc_samples=10):
        print("\nEvaluating model...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        all_uncertainties = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(test_loader, desc='Evaluation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                mean_probs, uncertainty = self.model.forward_with_uncertainty(images, num_samples=num_mc_samples)
                
                _, predicted = mean_probs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(mean_probs.cpu().numpy())
                all_uncertainties.extend(uncertainty.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_uncertainties = np.array(all_uncertainties)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        ece = compute_calibration_error(all_labels, all_probs)
        brier = compute_brier_score(all_labels, all_probs)
        
        metrics['ece'] = ece
        metrics['brier_score'] = brier
        metrics['mean_uncertainty'] = all_uncertainties.mean()
        
        return metrics, all_preds, all_labels, all_probs, all_uncertainties
    
    def evaluate_with_confidence_intervals(self, test_loader, num_bootstrap=100):
        print(f"\nComputing confidence intervals with {num_bootstrap} bootstrap samples...")
        
        _, _, all_labels, all_probs, _ = self.evaluate(test_loader, num_mc_samples=1)
        
        bootstrap_metrics = {
            'accuracy': [],
            'auc': [],
            'sensitivity': [],
            'specificity': []
        }
        
        n_samples = len(all_labels)
        
        for _ in tqdm(range(num_bootstrap), desc='Bootstrap'):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            
            boot_labels = all_labels[indices]
            boot_probs = all_probs[indices]
            boot_preds = np.argmax(boot_probs, axis=1)
            
            boot_metrics = compute_metrics(boot_labels, boot_preds, boot_probs)
            
            bootstrap_metrics['accuracy'].append(boot_metrics['accuracy'])
            bootstrap_metrics['auc'].append(boot_metrics['auc'])
            bootstrap_metrics['sensitivity'].append(boot_metrics['sensitivity'])
            if 'specificity' in boot_metrics:
                bootstrap_metrics['specificity'].append(boot_metrics['specificity'])
        
        ci_results = {}
        for metric_name, values in bootstrap_metrics.items():
            mean = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            ci_results[metric_name] = {
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper
            }
        
        return ci_results
    
    def save_results(self, metrics, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        results = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'sensitivity': float(metrics['sensitivity']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'ece': float(metrics['ece']),
            'brier_score': float(metrics['brier_score']),
            'mean_uncertainty': float(metrics['mean_uncertainty'])
        }
        
        if 'specificity' in metrics:
            results['specificity'] = float(metrics['specificity'])
        if 'ppv' in metrics:
            results['ppv'] = float(metrics['ppv'])
        if 'npv' in metrics:
            results['npv'] = float(metrics['npv'])
        
        if 'confusion_matrix' in metrics:
            results['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--output_path', type=str, default='../results/evaluation_results.json', 
                       help='Path to save evaluation results')
    parser.add_argument('--mc_samples', type=int, default=10, help='Number of MC dropout samples')
    parser.add_argument('--compute_ci', action='store_true', help='Compute confidence intervals')
    
    args = parser.parse_args()
    
    config = {
        'num_classes': 2,
        'img_size': 224,
        'base_channels': 32,
        'transformer_dim': 256,
        'nhead': 4,
        'num_transformer_layers': 2,
        'dropout': 0.3
    }
    
    print("Creating test dataloader...")
    _, _, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=4,
        img_size=config['img_size']
    )
    
    print(f"Test samples: {len(test_loader.dataset)}")
    
    evaluator = ModelEvaluator(args.model_path, config)
    
    metrics, preds, labels, probs, uncertainties = evaluator.evaluate(
        test_loader, 
        num_mc_samples=args.mc_samples
    )
    
    print_metrics(metrics, prefix="Test ")
    print(f"\nExpected Calibration Error: {metrics['ece']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Mean Uncertainty: {metrics['mean_uncertainty']:.4f}")
    
    if args.compute_ci:
        ci_results = evaluator.evaluate_with_confidence_intervals(test_loader, num_bootstrap=100)
        print("\n95% Confidence Intervals:")
        for metric_name, values in ci_results.items():
            print(f"  {metric_name}: {values['mean']:.4f} ({values['ci_lower']:.4f} - {values['ci_upper']:.4f})")
        
        metrics['confidence_intervals'] = ci_results
    
    evaluator.save_results(metrics, args.output_path)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
