# homeostatic_learner.py

import torch
import torch.nn.functional as F
from typing import NamedTuple

from torch import Tensor

# Define the PPO_Losses data structure locally to avoid import issues
class PPO_Losses(NamedTuple):
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    kl_loss: Tensor
    kl_div: Tensor

# --- Core Sample Factory imports for the Learner ---
from sample_factory.algo.learning.learner_worker import Learner
from sample_factory.utils.typing import Config

# VERIFICATION FIX: Implement GAE locally to remove the final import dependency.
# This makes our learner self-contained and robust to library version changes.
def calculate_gae_advantages(rewards: Tensor, dones: Tensor, values: Tensor, gamma: float, gae_lambda: float):
    """
    A standalone implementation of Generalized Advantage Estimation (GAE).
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = 0
    
    # The values tensor is expected to have one extra timestep for bootstrapping.
    # e.g., for a rollout of length T, values should be T+1 long.
    values = values.squeeze(-1)
    
    for t in reversed(range(rewards.shape)):
        # The 'done' mask is for the *next* state. If the next state is terminal,
        # its value is 0.
        next_is_not_done = 1.0 - dones[t].float()
        next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_is_not_done - values[t]
        advantages[t] = last_advantage = delta + gamma * gae_lambda * next_is_not_done * last_advantage

    # value_targets are the advantages plus the original value estimates
    value_targets = advantages + values[:-1] # Exclude the last value, which was for bootstrapping
    return advantages, value_targets


# Define a structure for our multiple advantage streams
class GaeResults(NamedTuple):
    advantages: Tensor
    value_targets: Tensor

class HomeostaticLearner(Learner):
    def __init__(self, cfg: Config, policy_versions_root, policy_version, ppo_clip_value):
        super().__init__(cfg, policy_versions_root, policy_version)
        self.ppo_clip_value = ppo_clip_value
        self.critic_streams = ["food", "drink", "energy", "health"]

    def _calc_gae(self, rewards, dones, values):
        """Helper to calculate GAE for a single stream using our local implementation."""
        advantages, value_targets = calculate_gae_advantages(rewards, dones, values, self.gamma, self.gae_lambda)
        return GaeResults(advantages=advantages, value_targets=value_targets)

    def _build_loss(self, mb, policy_outputs):
        action_dist, model_outputs, _ = policy_outputs
        values_dict = {
            "food": model_outputs["value_food"],
            "drink": model_outputs["value_drink"],
            "energy": model_outputs["value_energy"],
            "health": model_outputs["value_health"],
        }
        
        actions = mb.actions
        log_prob_actions = action_dist.log_prob(actions)
        
        # --- 1. Calculate ICM Loss and Intrinsic Reward ---
        phi_next, pred_phi_next, pred_action_logits = self.actor_critic.icm(
            mb.obs['obs'],
            mb.next_obs['obs'],
            mb.actions
        )

        phi_next = phi_next.detach()
        loss_f = 0.5 * F.mse_loss(pred_phi_next, phi_next, reduction='none').mean(dim=-1)
        loss_i = F.cross_entropy(pred_action_logits, actions.squeeze(-1).long(), reduction='none')
        intrinsic_reward = self.cfg.icm_reward_scale * loss_f.detach()
        
        # --- 2. Calculate Advantages for each of the 4 streams ---
        advantages_dict = {}
        value_targets_dict = {}
        
        for stream in self.critic_streams:
            extrinsic_rewards = mb.rewards[stream] 
            total_rewards = extrinsic_rewards + intrinsic_reward
            
            gae_results = self._calc_gae(total_rewards, mb.dones, values_dict[stream])
            advantages_dict[stream] = gae_results.advantages
            value_targets_dict[stream] = gae_results.value_targets

        # --- 3. Calculate Total PPO-style Loss ---
        total_advantages = torch.stack(list(advantages_dict.values()), dim=0).sum(dim=0)
        
        if self.cfg.normalize_advantages:
            total_advantages = (total_advantages - total_advantages.mean()) / (total_advantages.std() + 1e-8)

        # Policy Loss
        ratio = torch.exp(log_prob_actions - mb.log_prob_actions)
        surr1 = ratio * total_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_value, 1.0 + self.ppo_clip_value) * total_advantages
        policy_loss = -torch.min(surr1, surr2)

        # Value Loss
        total_value_loss = 0
        for stream in self.critic_streams:
            value_loss = F.mse_loss(values_dict[stream].squeeze(-1)[:-1], value_targets_dict[stream])
            total_value_loss += value_loss
        
        entropy = action_dist.entropy()
        entropy_loss = -self.entropy_coeff * entropy

        # --- 4. Combine all losses ---
        beta = self.cfg.icm_beta
        icm_loss = (1 - beta) * loss_i + beta * loss_f
        
        total_loss = (
            policy_loss.mean() + 
            self.value_loss_coeff * total_value_loss + 
            entropy_loss.mean() +
            self.cfg.icm_loss_coeff * icm_loss.mean()
        )

        return PPO_Losses(
            total_loss=total_loss,
            policy_loss=policy_loss.mean(),
            value_loss=total_value_loss,
            entropy_loss=entropy_loss.mean(),
            kl_loss=torch.tensor(0.0),
            kl_div=torch.tensor(0.0),
        )