import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class PPOAgent(nn.Module):
    def __init__(self, state_dim=100, action_dim=60, hidden_dim=256):
        super(PPOAgent, self).__init__()
        
        self.shared_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        shared_features = self.shared_network(state)
        
        action_logits = self.policy_head(shared_features)
        action_probs = F.softmax(action_logits, dim=-1)
        
        state_value = self.value_head(shared_features)
        
        return action_probs, state_value
    
    def get_action(self, state, deterministic=False):
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            log_prob = torch.log(action_probs.gather(1, action.unsqueeze(-1)))
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, state_value
    
    def evaluate_actions(self, states, actions):
        action_probs, state_values = self.forward(states)
        
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        return action_log_probs, state_values, dist_entropy


class PPOTrainer:
    """
    PPO Trainer with buffer management as per paper Table 4:
    - Buffer size: 1024 transitions
    - Batch size: 128 for PPO updates
    """
    def __init__(self, agent, actor_lr=1e-4, critic_lr=5e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, entropy_coef=0.02, 
                 value_loss_coef=0.5, max_grad_norm=0.5, buffer_size=1024):
        self.agent = agent
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.buffer_size = buffer_size  # Paper Table 4: Buffer Size 1024
        
        self.optimizer = torch.optim.Adam([
            {'params': agent.shared_network.parameters(), 'lr': actor_lr},
            {'params': agent.policy_head.parameters(), 'lr': actor_lr},
            {'params': agent.value_head.parameters(), 'lr': critic_lr}
        ])
        
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition with buffer size management."""
        # Convert to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.long)
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, dtype=torch.float32)
            
        self.buffer['states'].append(state.detach().cpu())
        self.buffer['actions'].append(action.detach().cpu() if action.requires_grad else action.cpu())
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob.detach().cpu() if isinstance(log_prob, torch.Tensor) and log_prob.requires_grad else (log_prob.cpu() if isinstance(log_prob, torch.Tensor) else log_prob))
        self.buffer['dones'].append(done)
        
        # Enforce buffer size limit (Paper Table 4)
        if len(self.buffer['states']) > self.buffer_size:
            for key in self.buffer:
                self.buffer[key] = self.buffer[key][-self.buffer_size:]
    
    def compute_gae(self, rewards, values, dones, next_value):
        advantages = []
        gae = 0
        
        values = values + [next_value]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        
        return advantages, returns
    
    def update(self, num_epochs=4):
        """
        PPO update with proper tensor handling.
        Paper Table 4: 4 PPO epochs per update.
        """
        if len(self.buffer['states']) == 0:
            return {}
        
        # Proper tensor conversion
        states = torch.stack(self.buffer['states'])
        actions = torch.stack([a if isinstance(a, torch.Tensor) else torch.tensor(a) for a in self.buffer['actions']])
        old_log_probs = torch.stack([lp if isinstance(lp, torch.Tensor) else torch.tensor(lp) for lp in self.buffer['log_probs']])
        rewards = self.buffer['rewards']
        values = self.buffer['values']
        dones = self.buffer['dones']
        
        # Move to device
        device = next(self.agent.parameters()).device
        states = states.to(device)
        actions = actions.to(device)
        old_log_probs = old_log_probs.to(device)
        
        with torch.no_grad():
            _, next_value = self.agent(states[-1].unsqueeze(0))
            next_value = next_value.squeeze().item() if next_value.numel() == 1 else next_value.squeeze()
        
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(states.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(states.device)
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(num_epochs):
            action_log_probs, state_values, dist_entropy = self.agent.evaluate_actions(states, actions)
            
            state_values = state_values.squeeze()
            
            ratio = torch.exp(action_log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(state_values, returns)
            
            entropy_loss = -dist_entropy.mean()
            
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += dist_entropy.mean().item()
        
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / num_epochs,
            'value_loss': total_value_loss / num_epochs,
            'entropy': total_entropy / num_epochs
        }
    
    def clear_buffer(self):
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }


if __name__ == "__main__":
    agent = PPOAgent(state_dim=100, action_dim=60, hidden_dim=256)
    print(f"PPO Agent created with {sum(p.numel() for p in agent.parameters())} parameters")
    
    dummy_state = torch.randn(1, 100)
    action, log_prob, value = agent.get_action(dummy_state)
    print(f"Action: {action.item()}, Log Prob: {log_prob.item():.4f}, Value: {value.item():.4f}")
