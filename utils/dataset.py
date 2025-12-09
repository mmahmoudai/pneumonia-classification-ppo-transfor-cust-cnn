import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

class ChestXRayDataset(Dataset):
    def __init__(self, data_dir, csv_file=None, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        if csv_file and os.path.exists(csv_file):
            self.data_frame = pd.read_csv(csv_file)
            self.image_paths = self.data_frame['image_path'].tolist()
            self.labels = self.data_frame['label'].tolist()
        else:
            self.image_paths, self.labels = self._load_from_directory()
        
    def _load_from_directory(self):
        image_paths = []
        labels = []
        
        class_names = ['NORMAL', 'PNEUMONIA']
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(self.data_dir, self.split, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_idx)
        
        return image_paths, labels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path


def get_transforms(img_size=224, augment=False):
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_dir, batch_size=64, num_workers=4, img_size=224, 
                       train_csv=None, val_csv=None, test_csv=None):
    train_transform = get_transforms(img_size=img_size, augment=False)
    val_transform = get_transforms(img_size=img_size, augment=False)
    
    train_dataset = ChestXRayDataset(
        data_dir=data_dir,
        csv_file=train_csv,
        transform=train_transform,
        split='train'
    )
    
    val_dataset = ChestXRayDataset(
        data_dir=data_dir,
        csv_file=val_csv,
        transform=val_transform,
        split='val'
    )
    
    test_dataset = ChestXRayDataset(
        data_dir=data_dir,
        csv_file=test_csv,
        transform=val_transform,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


class StateExtractor:
    def __init__(self):
        self.state_dim = 100
        
    def extract_state(self, images, model_outputs=None, training_metrics=None, curriculum_stage=0):
        batch_size = images.size(0)
        
        state_features = []
        
        global_stats = self._compute_global_stats(images)
        state_features.append(global_stats)
        
        lung_features = self._compute_lung_features(images)
        state_features.append(lung_features)
        
        if model_outputs is not None:
            uncertainty_features = self._compute_uncertainty_features(model_outputs)
        else:
            uncertainty_features = torch.zeros(24)
        state_features.append(uncertainty_features)
        
        if training_metrics is not None:
            dynamics_features = self._compute_dynamics_features(training_metrics, curriculum_stage)
        else:
            dynamics_features = torch.zeros(16)
        state_features.append(dynamics_features)
        
        state = torch.cat(state_features, dim=0)
        
        if state.size(0) < self.state_dim:
            padding = torch.zeros(self.state_dim - state.size(0))
            state = torch.cat([state, padding], dim=0)
        elif state.size(0) > self.state_dim:
            state = state[:self.state_dim]
        
        return state
    
    def _compute_global_stats(self, images):
        mean_intensity = images.mean(dim=[1, 2, 3]).mean()
        std_intensity = images.std(dim=[1, 2, 3]).mean()
        
        min_intensity = images.min()
        max_intensity = images.max()
        
        hist_features = torch.histc(images.flatten(), bins=10, min=0, max=1)
        hist_features = hist_features / hist_features.sum()
        
        edge_density = self._compute_edge_density(images)
        
        features = torch.cat([
            torch.tensor([mean_intensity, std_intensity, min_intensity, max_intensity, edge_density]),
            hist_features
        ])
        
        if features.size(0) < 32:
            padding = torch.zeros(32 - features.size(0))
            features = torch.cat([features, padding])
        else:
            features = features[:32]
        
        return features
    
    def _compute_edge_density(self, images):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        gray = images.mean(dim=1, keepdim=True)
        
        edges_x = torch.nn.functional.conv2d(gray, sobel_x, padding=1)
        edges_y = torch.nn.functional.conv2d(gray, sobel_y, padding=1)
        
        edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
        edge_density = edge_magnitude.mean()
        
        return edge_density
    
    def _compute_lung_features(self, images):
        """
        Compute 28-dimensional lung field features as per paper Section 3.2:
        - Area ratios (lung field coverage)
        - Centroid positions (left/right lung centers)
        - Boundary smoothness metrics
        """
        features = torch.zeros(28)
        
        # Convert to grayscale
        gray = images.mean(dim=1)  # [B, H, W]
        batch_size = gray.size(0)
        
        # Threshold-based lung segmentation (simplified)
        threshold = gray.mean()
        lung_mask = (gray < threshold).float()
        
        # === Area Ratios (features 0-7) ===
        # Total lung area ratio
        features[0] = lung_mask.mean()
        
        # Left/right lung area ratios (split image vertically)
        h, w = gray.shape[1], gray.shape[2]
        left_mask = lung_mask[:, :, :w//2]
        right_mask = lung_mask[:, :, w//2:]
        features[1] = left_mask.mean()
        features[2] = right_mask.mean()
        features[3] = (left_mask.mean() / (right_mask.mean() + 1e-8))  # Symmetry ratio
        
        # Quadrant area ratios
        features[4] = lung_mask[:, :h//2, :w//2].mean()  # Top-left
        features[5] = lung_mask[:, :h//2, w//2:].mean()  # Top-right
        features[6] = lung_mask[:, h//2:, :w//2].mean()  # Bottom-left
        features[7] = lung_mask[:, h//2:, w//2:].mean()  # Bottom-right
        
        # === Centroid Positions (features 8-15) ===
        # Compute centroids using moment calculation
        y_coords = torch.arange(h, dtype=torch.float32).view(1, h, 1).expand(batch_size, h, w)
        x_coords = torch.arange(w, dtype=torch.float32).view(1, 1, w).expand(batch_size, h, w)
        
        lung_sum = lung_mask.sum(dim=[1, 2]) + 1e-8
        centroid_y = (lung_mask * y_coords).sum(dim=[1, 2]) / lung_sum
        centroid_x = (lung_mask * x_coords).sum(dim=[1, 2]) / lung_sum
        
        features[8] = centroid_y.mean() / h  # Normalized Y centroid
        features[9] = centroid_x.mean() / w  # Normalized X centroid
        features[10] = centroid_y.std()  # Y centroid variance
        features[11] = centroid_x.std()  # X centroid variance
        
        # Left lung centroid
        left_sum = left_mask.sum(dim=[1, 2]) + 1e-8
        features[12] = (left_mask * y_coords[:, :, :w//2]).sum(dim=[1, 2]).mean() / (h * left_sum.mean())
        features[13] = (left_mask * x_coords[:, :, :w//2]).sum(dim=[1, 2]).mean() / (w * left_sum.mean())
        
        # Right lung centroid
        right_sum = right_mask.sum(dim=[1, 2]) + 1e-8
        features[14] = (right_mask * y_coords[:, :, w//2:]).sum(dim=[1, 2]).mean() / (h * right_sum.mean())
        features[15] = (right_mask * x_coords[:, :, w//2:]).sum(dim=[1, 2]).mean() / (w * right_sum.mean())
        
        # === Boundary Smoothness (features 16-23) ===
        # Compute boundary using morphological gradient approximation
        # Sobel edge detection on lung mask
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        lung_mask_4d = lung_mask.unsqueeze(1)  # [B, 1, H, W]
        boundary_x = torch.nn.functional.conv2d(lung_mask_4d, sobel_x, padding=1)
        boundary_y = torch.nn.functional.conv2d(lung_mask_4d, sobel_y, padding=1)
        boundary = torch.sqrt(boundary_x**2 + boundary_y**2)
        
        features[16] = boundary.mean()  # Mean boundary intensity
        features[17] = boundary.std()   # Boundary variance (smoothness indicator)
        features[18] = boundary.max()   # Max boundary intensity
        features[19] = (boundary > 0.5).float().mean()  # Boundary density
        
        # Boundary smoothness per quadrant
        features[20] = boundary[:, :, :h//2, :w//2].mean()
        features[21] = boundary[:, :, :h//2, w//2:].mean()
        features[22] = boundary[:, :, h//2:, :w//2].mean()
        features[23] = boundary[:, :, h//2:, w//2:].mean()
        
        # === Additional Intensity Features (features 24-27) ===
        features[24] = gray.mean()  # Mean intensity
        features[25] = gray.std()   # Intensity variance
        features[26] = (gray * lung_mask).sum() / (lung_mask.sum() + 1e-8)  # Mean lung intensity
        features[27] = ((gray * lung_mask)**2).sum() / (lung_mask.sum() + 1e-8) - features[26]**2  # Lung intensity variance
        
        return features
    
    def _compute_uncertainty_features(self, model_outputs):
        if isinstance(model_outputs, tuple):
            probs, uncertainty = model_outputs
        else:
            probs = torch.softmax(model_outputs, dim=1)
            uncertainty = torch.zeros_like(probs)
        
        confidence = probs.max(dim=1)[0]
        
        bins = torch.linspace(0, 1, 11)
        hist = torch.histc(confidence, bins=10, min=0, max=1)
        hist = hist / (hist.sum() + 1e-8)
        
        features = torch.cat([
            confidence.mean().unsqueeze(0),
            confidence.std().unsqueeze(0),
            uncertainty.mean().unsqueeze(0),
            hist
        ])
        
        if features.size(0) < 24:
            padding = torch.zeros(24 - features.size(0))
            features = torch.cat([features, padding])
        else:
            features = features[:24]
        
        return features
    
    def _compute_dynamics_features(self, training_metrics, curriculum_stage):
        """
        Compute 16-dimensional training dynamics features (Section 3.2):
        - Loss trajectory statistics
        - Gradient magnitudes
        - Curriculum stage encoding
        - Learning progress indicators
        """
        features = torch.zeros(16)
        
        # Current metrics (features 0-3)
        if 'loss' in training_metrics:
            features[0] = training_metrics['loss']
        if 'accuracy' in training_metrics:
            features[1] = training_metrics['accuracy']
        if 'gradient_norm' in training_metrics:
            features[2] = min(training_metrics['gradient_norm'], 10.0) / 10.0  # Normalized
        features[3] = curriculum_stage / 5.0  # Normalized curriculum stage
        
        # Loss trajectory statistics (features 4-7)
        if 'loss_history' in training_metrics and len(training_metrics['loss_history']) > 0:
            loss_hist = training_metrics['loss_history']
            features[4] = np.mean(loss_hist)  # Mean loss
            features[5] = np.std(loss_hist) if len(loss_hist) > 1 else 0  # Loss variance
            features[6] = loss_hist[-1] - loss_hist[0] if len(loss_hist) > 1 else 0  # Loss trend
            features[7] = min(loss_hist) if loss_hist else 0  # Min loss achieved
        
        # Accuracy trajectory statistics (features 8-11)
        if 'acc_history' in training_metrics and len(training_metrics['acc_history']) > 0:
            acc_hist = training_metrics['acc_history']
            features[8] = np.mean(acc_hist)  # Mean accuracy
            features[9] = np.std(acc_hist) if len(acc_hist) > 1 else 0  # Accuracy variance
            features[10] = acc_hist[-1] - acc_hist[0] if len(acc_hist) > 1 else 0  # Accuracy trend
            features[11] = max(acc_hist) if acc_hist else 0  # Max accuracy achieved
        
        # Learning progress indicators (features 12-15)
        features[12] = 1.0 if features[6] < 0 else 0.0  # Is loss decreasing?
        features[13] = 1.0 if features[10] > 0 else 0.0  # Is accuracy increasing?
        features[14] = min(1.0, features[2] * 2)  # Gradient magnitude indicator
        features[15] = curriculum_stage / 4.0  # Alternative curriculum encoding
        
        return features


if __name__ == "__main__":
    print("Dataset module loaded successfully")
    print("To use: create_dataloaders(data_dir='path/to/data', batch_size=64)")
