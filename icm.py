import torch
import torch.nn as nn
import torch.nn.functional as F

from hcrafter_model import HcrafterEncoder  # Reuse your existing encoder

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, cfg, obs_space, action_space):
        super().__init__()
        # Use your existing HcrafterEncoder to get feature representations
        self.feature_encoder = HcrafterEncoder(cfg, obs_space)
        feature_size = self.feature_encoder.get_out_size()
        num_actions = action_space.n

        # Inverse Model: Predicts action from (state features, next state features)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        # Forward Model: Predicts next state features from (state features, action)
        # We need to one-hot encode the action
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + num_actions, 512),
            nn.ReLU(),
            nn.Linear(512, feature_size)
        )

    def forward(self, obs_dict, next_obs_dict, action):
        # Encode observations into the learned feature space
        phi_state = self.feature_encoder(obs_dict)
        phi_next_state = self.feature_encoder(next_obs_dict)

        # Predict next state features for the forward loss and reward
        action_one_hot = F.one_hot(action, num_classes=self.inverse_model[-1].out_features).float()
        pred_phi_next_state = self.forward_model(torch.cat([phi_state, action_one_hot], dim=1))
        
        # Predict the action for the inverse loss
        pred_action_logits = self.inverse_model(torch.cat([phi_state, phi_next_state], dim=1))

        return pred_phi_next_state, phi_next_state, pred_action_logits

    def get_reward(self, obs_dict, next_obs_dict, action):
        with torch.no_grad():
            pred_phi_next_state, phi_next_state, _ = self.forward(obs_dict, next_obs_dict, action)
            # Curiosity reward is the mean squared error between predicted and actual next state features
            reward = F.mse_loss(pred_phi_next_state, phi_next_state, reduction='none').mean(dim=1)
        return reward

    def get_loss(self, obs_dict, next_obs_dict, action):
        pred_phi_next_state, phi_next_state, pred_action_logits = self.forward(obs_dict, next_obs_dict, action)
        
        forward_loss = F.mse_loss(pred_phi_next_state, phi_next_state)
        inverse_loss = F.cross_entropy(pred_action_logits, action.long())
        
        return forward_loss, inverse_loss