import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image
import random

class ClinicallyConstrainedAugmentation:
    """
    Clinically constrained augmentation pipeline as per paper Section 3.3.
    
    Augmentation constraints derived from radiological literature [27,28]:
    - Rotation: θ ∈ [-8°, 8°] - Mimics projection variability from patient positioning
    - Brightness: β ∈ [0.9, 1.2] - Preserves lung field visibility across exposure settings
    - Contrast: γ ∈ [0.8, 1.2] - Maintains pathological opacity visibility
    - Gaussian Noise: σ ∈ [0, 0.03] - Realistic acquisition noise without obscuring pathology
    - Gaussian Blur: σ ∈ [0, 1.0] - Minor motion artifacts preserving anatomical detail
    
    Note: Horizontal flipping is explicitly EXCLUDED as it inverts anatomical laterality.
    """
    def __init__(self):
        # Clinical constraints from Table 1 in paper
        self.augmentation_types = {
            'rotation': {'range': (-8, 8), 'levels': 12, 'clinical_rationale': 'Patient positioning variability'},
            'brightness': {'range': (0.9, 1.2), 'levels': 12, 'clinical_rationale': 'Exposure settings variation'},
            'contrast': {'range': (0.8, 1.2), 'levels': 12, 'clinical_rationale': 'Pathological opacity visibility'},
            'gaussian_noise': {'range': (0.0, 0.03), 'levels': 12, 'clinical_rationale': 'Acquisition noise simulation'},
            'gaussian_blur': {'range': (0.0, 1.0), 'levels': 12, 'clinical_rationale': 'Minor motion artifacts'}
        }
        
        # Total action space: 5 types × 12 levels = 60 discrete actions
        self.action_space_size = sum(aug['levels'] for aug in self.augmentation_types.values())
        
        self.action_to_aug = self._build_action_mapping()
        
        # Curriculum difficulty scaling parameter (Section 3.5)
        self.current_difficulty = 0.0
        self.alpha_adapt = 0.5  # Adaptation time constant
        
    def _build_action_mapping(self):
        action_mapping = []
        
        for aug_type, params in self.augmentation_types.items():
            min_val, max_val = params['range']
            levels = params['levels']
            
            for level in range(levels):
                intensity = min_val + (max_val - min_val) * (level / (levels - 1))
                action_mapping.append({
                    'type': aug_type,
                    'intensity': intensity
                })
        
        return action_mapping
    
    def set_difficulty(self, difficulty):
        """
        Set curriculum difficulty level (Section 3.5).
        d_k ∈ [0, 1] where d_k = (k-1)/(K-1) for K=5 stages.
        """
        self.current_difficulty = difficulty
    
    def _scale_intensity(self, base_intensity, aug_type):
        """
        Scale augmentation intensity based on curriculum difficulty (Eq. 7 in paper).
        τ_t = τ_base · (1 + d_k · α_adapt)
        """
        if aug_type in ['rotation', 'gaussian_noise', 'gaussian_blur']:
            # For these, higher difficulty = more intense augmentation
            scaled = base_intensity * (1 + self.current_difficulty * self.alpha_adapt)
        elif aug_type == 'brightness':
            # Scale deviation from 1.0 (neutral brightness)
            deviation = base_intensity - 1.0
            scaled = 1.0 + deviation * (1 + self.current_difficulty * self.alpha_adapt)
        elif aug_type == 'contrast':
            # Scale deviation from 1.0 (neutral contrast)
            deviation = base_intensity - 1.0
            scaled = 1.0 + deviation * (1 + self.current_difficulty * self.alpha_adapt)
        else:
            scaled = base_intensity
        return scaled
    
    def apply_augmentation(self, image, action, use_difficulty_scaling=True):
        """
        Apply clinically constrained augmentation to image.
        
        Args:
            image: Input image (torch.Tensor or PIL.Image)
            action: Action index (0-59) selecting augmentation type and intensity
            use_difficulty_scaling: Whether to scale intensity by curriculum difficulty
        
        Returns:
            Augmented image as torch.Tensor
        """
        if isinstance(image, torch.Tensor):
            image = TF.to_pil_image(image)
        
        aug_params = self.action_to_aug[action]
        aug_type = aug_params['type']
        base_intensity = aug_params['intensity']
        
        # Apply curriculum difficulty scaling (Section 3.5)
        if use_difficulty_scaling:
            intensity = self._scale_intensity(base_intensity, aug_type)
        else:
            intensity = base_intensity
        
        # Ensure intensity stays within clinical bounds
        intensity = self._clip_to_clinical_bounds(intensity, aug_type)
        
        if aug_type == 'rotation':
            image = TF.rotate(image, angle=intensity, fill=0)
        
        elif aug_type == 'brightness':
            image = TF.adjust_brightness(image, brightness_factor=intensity)
        
        elif aug_type == 'contrast':
            image = TF.adjust_contrast(image, contrast_factor=intensity)
        
        elif aug_type == 'gaussian_noise':
            image_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(0, intensity, image_array.shape)
            noisy_image = np.clip(image_array + noise, 0, 1)
            image = Image.fromarray((noisy_image * 255).astype(np.uint8))
        
        elif aug_type == 'gaussian_blur':
            if intensity > 0:
                kernel_size = int(2 * np.ceil(2 * intensity) + 1)
                kernel_size = max(3, kernel_size)  # Minimum kernel size
                if kernel_size % 2 == 0:
                    kernel_size += 1  # Ensure odd kernel size
                image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=intensity)
        
        if isinstance(image, Image.Image):
            image = TF.to_tensor(image)
        
        return image
    
    def _clip_to_clinical_bounds(self, intensity, aug_type):
        """Ensure augmentation intensity remains within clinically valid bounds."""
        bounds = self.augmentation_types[aug_type]['range']
        return np.clip(intensity, bounds[0], bounds[1])
    
    def get_random_action(self):
        return random.randint(0, self.action_space_size - 1)
    
    def is_clinically_valid(self, image, augmented_image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if isinstance(augmented_image, torch.Tensor):
            augmented_image = augmented_image.cpu().numpy()
        
        mean_diff = np.abs(image.mean() - augmented_image.mean())
        if mean_diff > 0.3:
            return False
        
        std_ratio = augmented_image.std() / (image.std() + 1e-8)
        if std_ratio < 0.5 or std_ratio > 2.0:
            return False
        
        return True
    
    def compute_implausibility_penalty(self, image, augmented_image):
        if not self.is_clinically_valid(image, augmented_image):
            return 1.0
        return 0.0


class CurriculumScheduler:
    def __init__(self, num_stages=5, initial_difficulty=0.0):
        self.num_stages = num_stages
        self.current_stage = 0
        self.difficulty_levels = [i / (num_stages - 1) for i in range(num_stages)]
        self.current_difficulty = initial_difficulty
        
        self.stage_ece_history = []
        self.safety_threshold = 1.5
        
    def get_difficulty(self):
        return self.difficulty_levels[self.current_stage]
    
    def advance_stage(self, current_ece):
        self.stage_ece_history.append(current_ece)
        
        if len(self.stage_ece_history) > 1:
            prev_ece = self.stage_ece_history[-2]
            if current_ece > self.safety_threshold * prev_ece:
                print(f"Safety rollback triggered: ECE increased from {prev_ece:.4f} to {current_ece:.4f}")
                if self.current_stage > 0:
                    self.current_stage -= 1
                return False
        
        if self.current_stage < self.num_stages - 1:
            self.current_stage += 1
            print(f"Advanced to curriculum stage {self.current_stage + 1}/{self.num_stages}")
            return True
        
        return False
    
    def get_stage_info(self):
        return {
            'stage': self.current_stage + 1,
            'total_stages': self.num_stages,
            'difficulty': self.get_difficulty(),
            'ece_history': self.stage_ece_history
        }


def create_augmentation_pipeline():
    return ClinicallyConstrainedAugmentation()


def create_curriculum_scheduler(num_stages=5):
    return CurriculumScheduler(num_stages=num_stages)


if __name__ == "__main__":
    aug = create_augmentation_pipeline()
    print(f"Total action space size: {aug.action_space_size}")
    print(f"Number of augmentation types: {len(aug.augmentation_types)}")
    
    dummy_image = torch.randn(3, 224, 224)
    action = aug.get_random_action()
    augmented = aug.apply_augmentation(dummy_image, action)
    print(f"Applied action {action}, output shape: {augmented.shape}")
    
    scheduler = create_curriculum_scheduler()
    print(f"Initial difficulty: {scheduler.get_difficulty()}")
