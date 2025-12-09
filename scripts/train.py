import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import json
from datetime import datetime

from models.cnn_transformer import create_model
from models.ppo_agent import PPOAgent, PPOTrainer
from utils.dataset import create_dataloaders, StateExtractor
from utils.augmentation import create_augmentation_pipeline, create_curriculum_scheduler
from utils.metrics import compute_metrics, compute_calibration_error


def set_seed(seed):
    """Set random seeds for reproducibility (Section 4.3: seeds 42, 123, 2025, 314, 555)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RLAugmentationTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set seed for reproducibility
        if 'seed' in config:
            set_seed(config['seed'])
            print(f"Random seed set to: {config['seed']}")
        
        print(f"Using device: {self.device}")
        
        self.classifier = create_model(
            num_classes=config['num_classes'],
            img_size=config['img_size'],
            base_channels=config['base_channels'],
            transformer_dim=config['transformer_dim'],
            nhead=config['nhead'],
            num_transformer_layers=config['num_transformer_layers'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.ppo_agent = PPOAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            hidden_dim=config['ppo_hidden_dim']
        ).to(self.device)
        
        self.classifier_optimizer = optim.AdamW(
            self.classifier.parameters(),
            lr=config['classifier_lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),  # Paper Section 4.3: default beta values
            eps=1e-8
        )
        
        # Learning rate scheduler: Cosine annealing with 5-epoch warm-up (Section 4.3)
        warmup_epochs = 5
        warmup_scheduler = LinearLR(
            self.classifier_optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=warmup_epochs
        )
        cosine_scheduler = CosineAnnealingLR(
            self.classifier_optimizer, 
            T_max=config['num_epochs'] - warmup_epochs,
            eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.classifier_optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )
        
        self.ppo_trainer = PPOTrainer(
            agent=self.ppo_agent,
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            entropy_coef=config['entropy_coef']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.augmentation = create_augmentation_pipeline()
        self.curriculum = create_curriculum_scheduler(num_stages=config['curriculum_stages'])
        self.state_extractor = StateExtractor()
        
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        self.best_val_acc = 0.0
        self.global_step = 0
        
        # Reward weights from paper Section 3.4
        self.reward_weights = {
            'alpha': config['reward_alpha'],   # Accuracy improvement weight
            'beta': config['reward_beta'],     # Diversity (entropy) weight  
            'delta': config['reward_delta'],   # Calibration improvement weight
            'gamma': config['reward_gamma'],   # Computational cost penalty
            'epsilon': config['reward_epsilon'] # Clinical implausibility penalty
        }
        
        # EMA for reward computation (Section 4.3 - λ = 0.99)
        self.ema_lambda = 0.99
        self.ema_acc = 0.0
        self.ema_ece = 0.0
        self.prev_auc = 0.0
        self.prev_ece = 1.0
        
        # Training dynamics for state representation
        self.current_training_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'gradient_norm': 0.0,
            'loss_history': [],
            'acc_history': []
        }
        
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_auc': [],
            'val_ece': []
        }
    
    def compute_reward(self, acc_improvement, diversity, calibration_improvement, 
                      computational_cost, implausibility_penalty):
        """
        Multi-objective reward function (Eq. 5 in paper):
        R_t = α·ΔAcc_t + β·ΔH(π_t) + δ·ΔECE_t - γ·C_t - ε·P_t
        
        Where:
        - ΔAcc_t: Accuracy/AUC improvement (EMA smoothed)
        - ΔH(π_t): Policy entropy (encourages diverse augmentation selection)
        - ΔECE_t: Calibration improvement (ECE decrease is positive)
        - C_t: Computational cost penalty
        - P_t: Clinical implausibility penalty
        """
        reward = (
            self.reward_weights['alpha'] * acc_improvement +
            self.reward_weights['beta'] * diversity +
            self.reward_weights['delta'] * calibration_improvement -
            self.reward_weights['gamma'] * computational_cost -
            self.reward_weights['epsilon'] * implausibility_penalty
        )
        return reward
    
    def update_ema_metrics(self, current_auc, current_ece):
        """
        Update EMA metrics for reward computation (Section 4.3).
        ΔAcc(t) = λ × ΔAcc(t-1) + (1-λ) × (AUC(t) - AUC(t-1))
        ΔECE(t) = λ × ΔECE(t-1) + (1-λ) × (ECE(t-1) - ECE(t))
        """
        # Compute instantaneous changes
        delta_auc = current_auc - self.prev_auc
        delta_ece = self.prev_ece - current_ece  # Decrease in ECE is positive
        
        # Apply EMA smoothing
        self.ema_acc = self.ema_lambda * self.ema_acc + (1 - self.ema_lambda) * delta_auc
        self.ema_ece = self.ema_lambda * self.ema_ece + (1 - self.ema_lambda) * delta_ece
        
        # Update previous values
        self.prev_auc = current_auc
        self.prev_ece = current_ece
        
        return self.ema_acc, self.ema_ece
    
    def train_epoch(self, train_loader, epoch, use_rl=True):
        """
        Train one epoch with RL-based adaptive augmentation.
        
        Training procedure (Section 4.3):
        1. Classifier pre-trained for 10 epochs without RL augmentation
        2. PPO agent activated for 140 additional epochs
        3. Agent selects one augmentation per image per batch
        4. PPO updates every 4 classifier batches (512 images)
        """
        self.classifier.train()
        self.ppo_agent.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        prev_acc = 0.0
        
        # Update augmentation difficulty based on curriculum stage (Section 3.5)
        current_difficulty = self.curriculum.get_difficulty()
        self.augmentation.set_difficulty(current_difficulty)
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            if use_rl and epoch >= self.config['rl_start_epoch']:
                # Get model uncertainty estimates for state (Section 3.2)
                with torch.no_grad():
                    model_outputs = self.classifier.forward_with_uncertainty(images, num_samples=5)
                
                # Extract modality-aware state representation (Section 3.2)
                # 100-dim: 32 image stats + 28 lung features + 24 uncertainty + 16 dynamics
                state = self.state_extractor.extract_state(
                    images,
                    model_outputs=model_outputs,
                    training_metrics=self.current_training_metrics,
                    curriculum_stage=self.curriculum.current_stage
                ).to(self.device)
                
                # PPO agent selects augmentation action
                action, log_prob, value = self.ppo_agent.get_action(state.unsqueeze(0))
                action = action.item()
                
                # Apply clinically constrained augmentation with difficulty scaling
                augmented_images = []
                for img in images:
                    aug_img = self.augmentation.apply_augmentation(
                        img.cpu(), action, use_difficulty_scaling=True
                    )
                    augmented_images.append(aug_img)
                augmented_images = torch.stack(augmented_images).to(self.device)
                
                # Compute clinical implausibility penalty (Section 3.4)
                implausibility = self.augmentation.compute_implausibility_penalty(
                    images[0].cpu(), augmented_images[0].cpu()
                )
            else:
                augmented_images = images
                implausibility = 0.0
            
            self.classifier_optimizer.zero_grad()
            outputs = self.classifier(augmented_images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            
            # Compute gradient norm for state representation
            grad_norm = 0.0
            for p in self.classifier.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
            
            self.classifier_optimizer.step()
            
            # Update training dynamics for state representation
            self.current_training_metrics['loss'] = loss.item()
            self.current_training_metrics['gradient_norm'] = grad_norm
            self.current_training_metrics['loss_history'].append(loss.item())
            if len(self.current_training_metrics['loss_history']) > 100:
                self.current_training_metrics['loss_history'] = self.current_training_metrics['loss_history'][-100:]
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            
            current_acc = 100. * correct / total
            
            # Update accuracy in training metrics
            self.current_training_metrics['accuracy'] = current_acc / 100.0
            self.current_training_metrics['acc_history'].append(current_acc / 100.0)
            if len(self.current_training_metrics['acc_history']) > 100:
                self.current_training_metrics['acc_history'] = self.current_training_metrics['acc_history'][-100:]
            
            if use_rl and epoch >= self.config['rl_start_epoch']:
                # Compute reward components (Section 3.4)
                
                # 1. Accuracy improvement using EMA (ΔAcc_t)
                acc_improvement = (current_acc - prev_acc) / 100.0
                
                # 2. Policy entropy for diversity (H(π_t) = -Σ p_k log(p_k))
                action_probs, _ = self.ppo_agent(state.unsqueeze(0))
                diversity = -(action_probs * torch.log(action_probs + 1e-8)).sum().item()
                
                # 3. Calibration improvement (use EMA values)
                calibration_improvement = self.ema_ece  # Pre-computed from validation
                
                # 4. Computational cost (normalized)
                computational_cost = 0.01
                
                # 5. Clinical implausibility penalty (P_t)
                # Already computed above
                
                # Compute total reward (Eq. 5)
                reward = self.compute_reward(
                    acc_improvement, diversity, calibration_improvement,
                    computational_cost, implausibility
                )
                
                # Store transition in PPO buffer
                self.ppo_trainer.store_transition(
                    state, torch.tensor(action), reward,
                    value.item(), log_prob.item(), False
                )
                
                # PPO update every 4 batches (Section 4.3)
                if (batch_idx + 1) % self.config['ppo_update_freq'] == 0:
                    ppo_losses = self.ppo_trainer.update(num_epochs=self.config['ppo_epochs'])
                    if ppo_losses:
                        self.writer.add_scalar('PPO/policy_loss', ppo_losses['policy_loss'], self.global_step)
                        self.writer.add_scalar('PPO/value_loss', ppo_losses['value_loss'], self.global_step)
                        self.writer.add_scalar('PPO/entropy', ppo_losses['entropy'], self.global_step)
                        self.writer.add_scalar('PPO/reward', reward, self.global_step)
                
                prev_acc = current_acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.2f}%'
            })
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        self.classifier.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc='Validation'):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.classifier(images)
                loss = self.criterion(outputs, labels)
                
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                total_loss += loss.item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100. * correct / total
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        metrics = compute_metrics(all_labels, all_preds, all_probs)
        ece = compute_calibration_error(all_labels, all_probs)
        
        return avg_loss, avg_acc, metrics, ece
    
    def train(self, train_loader, val_loader):
        print(f"\nStarting training for {self.config['num_epochs']} epochs")
        print(f"RL augmentation will start from epoch {self.config['rl_start_epoch']}")
        
        for epoch in range(self.config['num_epochs']):
            use_rl = epoch >= self.config['rl_start_epoch']
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch, use_rl=use_rl)
            
            val_loss, val_acc, val_metrics, val_ece = self.validate(val_loader)
            
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['train_acc'].append(train_acc)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['val_acc'].append(val_acc)
            self.metrics_history['val_auc'].append(val_metrics['auc'])
            self.metrics_history['val_ece'].append(val_ece)
            
            # Update EMA metrics for reward computation
            ema_acc, ema_ece = self.update_ema_metrics(val_metrics['auc'], val_ece)
            
            self.writer.add_scalar('Train/loss', train_loss, epoch)
            self.writer.add_scalar('Train/accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/loss', val_loss, epoch)
            self.writer.add_scalar('Val/accuracy', val_acc, epoch)
            self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
            self.writer.add_scalar('Val/ECE', val_ece, epoch)
            self.writer.add_scalar('EMA/accuracy', ema_acc, epoch)
            self.writer.add_scalar('EMA/calibration', ema_ece, epoch)
            self.writer.add_scalar('Curriculum/difficulty', self.curriculum.get_difficulty(), epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val AUC: {val_metrics['auc']:.4f}, Val ECE: {val_ece:.4f}")
            print(f"EMA ΔAcc: {ema_acc:.6f}, EMA ΔECE: {ema_ece:.6f}")
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, 'best_model.pth')
                print(f"New best model saved with accuracy: {val_acc:.2f}%")
            
            if (epoch + 1) % self.config['curriculum_update_freq'] == 0 and use_rl:
                self.curriculum.advance_stage(val_ece)
                curriculum_info = self.curriculum.get_stage_info()
                print(f"Curriculum Stage: {curriculum_info['stage']}/{curriculum_info['total_stages']}, "
                      f"Difficulty: {curriculum_info['difficulty']:.2f}")
            
            if (epoch + 1) % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Step learning rate scheduler (Section 4.3: cosine annealing)
            self.scheduler.step()
            current_lr = self.classifier_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('LR/classifier', current_lr, epoch)
        
        self.save_metrics()
        print("\nTraining completed!")
    
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': self.classifier.state_dict(),
            'ppo_agent_state_dict': self.ppo_agent.state_dict(),
            'classifier_optimizer_state_dict': self.classifier_optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics_history': self.metrics_history
        }
        
        save_path = os.path.join(self.config['checkpoint_dir'], filename)
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def save_metrics(self):
        metrics_path = os.path.join(self.config['checkpoint_dir'], 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
        print(f"Metrics saved to {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Train CNN-Transformer with RL-based Augmentation')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (paper uses: 42, 123, 2025, 314, 555)')
    parser.add_argument('--checkpoint_dir', type=str, default='../results/checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='../results/logs', help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    config = {
        'seed': args.seed,  # Reproducibility (Section 4.3)
        'num_classes': 2,
        'img_size': 224,
        'base_channels': 32,
        'transformer_dim': 256,
        'nhead': 4,
        'num_transformer_layers': 2,
        'dropout': 0.3,
        'state_dim': 100,
        'action_dim': 60,
        'ppo_hidden_dim': 256,
        'classifier_lr': args.lr,
        'actor_lr': 1e-4,
        'critic_lr': 5e-4,
        'weight_decay': 0.01,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_epsilon': 0.2,
        'entropy_coef': 0.02,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'rl_start_epoch': 10,
        'curriculum_stages': 5,
        'curriculum_update_freq': 20,
        'ppo_update_freq': 4,
        'ppo_epochs': 4,
        'save_freq': 10,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'reward_alpha': 1.0,
        'reward_beta': 0.3,
        'reward_delta': 0.5,
        'reward_gamma': 0.1,
        'reward_epsilon': 2.0
    }
    
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size'],
        num_workers=4,
        img_size=config['img_size']
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    trainer = RLAugmentationTrainer(config)
    
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
