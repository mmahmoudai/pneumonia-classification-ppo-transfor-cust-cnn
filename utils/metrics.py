import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc
)
import torch


def compute_metrics(y_true, y_pred, y_probs=None):
    metrics = {}
    
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
    metrics['sensitivity'] = metrics['recall']
    metrics['f1'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    if y_probs is not None:
        if y_probs.ndim == 2:
            y_probs_positive = y_probs[:, 1]
        else:
            y_probs_positive = y_probs
        
        try:
            metrics['auc'] = roc_auc_score(y_true, y_probs_positive)
        except:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    
    metrics['confusion_matrix'] = cm
    
    return metrics


def compute_calibration_error(y_true, y_probs, num_bins=10):
    if y_probs.ndim == 2:
        y_probs_positive = y_probs[:, 1]
    else:
        y_probs_positive = y_probs
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_probs_positive > bin_lower) & (y_probs_positive <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_probs_positive[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_brier_score(y_true, y_probs):
    if y_probs.ndim == 2:
        y_probs_positive = y_probs[:, 1]
    else:
        y_probs_positive = y_probs
    
    brier_score = np.mean((y_probs_positive - y_true) ** 2)
    return brier_score


def compute_confidence_intervals(metric_values, confidence=0.95):
    n = len(metric_values)
    mean = np.mean(metric_values)
    std = np.std(metric_values, ddof=1)
    
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin = t_value * (std / np.sqrt(n))
    
    ci_lower = mean - margin
    ci_upper = mean + margin
    
    return mean, ci_lower, ci_upper


def compute_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
    
    return cohens_d


def print_metrics(metrics, prefix=""):
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Precision:   {metrics['precision']:.4f}")
    print(f"  Recall:      {metrics['recall']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    if 'specificity' in metrics:
        print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  F1 Score:    {metrics['f1']:.4f}")
    if 'auc' in metrics:
        print(f"  AUC:         {metrics['auc']:.4f}")
    if 'ppv' in metrics:
        print(f"  PPV:         {metrics['ppv']:.4f}")
    if 'npv' in metrics:
        print(f"  NPV:         {metrics['npv']:.4f}")
    
    if 'confusion_matrix' in metrics:
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])


class MetricsTracker:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, preds, labels, probs=None):
        self.all_preds.extend(preds.cpu().numpy() if torch.is_tensor(preds) else preds)
        self.all_labels.extend(labels.cpu().numpy() if torch.is_tensor(labels) else labels)
        
        if probs is not None:
            self.all_probs.extend(probs.cpu().numpy() if torch.is_tensor(probs) else probs)
    
    def compute(self):
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        all_probs = np.array(self.all_probs) if len(self.all_probs) > 0 else None
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        
        if all_probs is not None:
            ece = compute_calibration_error(all_labels, all_probs)
            brier = compute_brier_score(all_labels, all_probs)
            metrics['ece'] = ece
            metrics['brier_score'] = brier
        
        return metrics


if __name__ == "__main__":
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    y_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8],
                        [0.85, 0.15], [0.6, 0.4], [0.1, 0.9], [0.95, 0.05]])
    
    metrics = compute_metrics(y_true, y_pred, y_probs)
    print_metrics(metrics)
    
    ece = compute_calibration_error(y_true, y_probs)
    print(f"\nExpected Calibration Error: {ece:.4f}")
