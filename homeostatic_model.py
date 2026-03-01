# homeostatic_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.utils.typing import Config, ObsSpace, ActionSpace

# ==================================================
# Section 1: Shared Backbone Components
# ==================================================

class HomeostaticEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        
        # VERIFICATION FIX: Extract the integer value for channels and features from the shape tuples.
        image_in_channels = obs_space['obs'].shape
        num_features = obs_space['measurements'].shape
        print("-----num_features:----", num_features)

        self.pov_encoder = nn.Sequential(
            # VERIFICATION FIX: Pass the integer 'image_in_channels' to the Conv2d layer.
            nn.Conv2d(image_in_channels[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_pov = torch.zeros(1, *obs_space['obs'].shape)
            cnn_out_size = self.pov_encoder(dummy_pov).shape[1]
            
        self.features_encoder = nn.Sequential(
            # VERIFICATION FIX: Pass the integer 'num_features' to the Linear layer.
            nn.Linear(num_features[0], 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self._out_size = cnn_out_size + 64

    def forward(self, obs_dict):
        pov_obs = obs_dict['obs'].float() / 255.0
        features_obs = obs_dict['measurements'].float()
        encoded_pov = self.pov_encoder(pov_obs)
        encoded_features = self.features_encoder(features_obs)
        return torch.cat([encoded_pov, encoded_features], dim=1)

    def get_out_size(self) -> int:
        return self._out_size

class HomeostaticCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        self.core_net = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU())
        self._out_size = 512

    def forward(self, head_output, rnn_states):
        core_output = self.core_net(head_output)
        return core_output, rnn_states
    
    def get_out_size(self) -> int:
        return self._out_size

# ==================================================
# Section 2: Intrinsic Curiosity Module (ICM)
# ==================================================

class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace):
        super().__init__()
        self.cfg = cfg
        
        # VERIFICATION FIX: Extract the integer value for channels from the shape tuple.
        image_in_channels = obs_space['obs'].shape
        
        self.feature_output_size = 288
        self.action_size = action_space.n
        self.feature_encoder = nn.Sequential(
            # VERIFICATION FIX: Pass the integer 'image_in_channels' to the Conv2d layer.
            nn.Conv2d(image_in_channels[0], 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1), nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy_pov = torch.zeros(1, *obs_space['obs'].shape)
            cnn_out_size = self.feature_encoder(dummy_pov).shape[1]
        self.feature_fc = nn.Linear(cnn_out_size, self.feature_output_size)
        self.inverse_model = nn.Sequential(
            nn.Linear(self.feature_output_size * 2, 256), nn.ReLU(),
            nn.Linear(256, self.action_size)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(self.feature_output_size + self.action_size, 256), nn.ReLU(),
            nn.Linear(256, self.feature_output_size)
        )

    def forward(self, state_pov, next_state_pov, action):
        state_pov = state_pov.float() / 255.0
        next_state_pov = next_state_pov.float() / 255.0
        phi_state = self.feature_fc(self.feature_encoder(state_pov))
        phi_next_state = self.feature_fc(self.feature_encoder(next_state_pov))
        action_one_hot = F.one_hot(action.squeeze(-1).long(), num_classes=self.action_size).float()
        forward_input = torch.cat([phi_state, action_one_hot], dim=1)
        predicted_phi_next_state = self.forward_model(forward_input)
        inverse_input = torch.cat([phi_state, phi_next_state], dim=1)
        predicted_action_logits = self.inverse_model(inverse_input)
        return phi_next_state, predicted_phi_next_state, predicted_action_logits

# ==================================================
# Section 3: Main Actor-Critic Class
# ==================================================

class HomeostaticActorCritic(ActorCritic):
    def __init__(self, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace):
        super().__init__(obs_space, action_space, cfg)
        self.obs_space = obs_space

        self.encoder = HomeostaticEncoder(cfg, self.obs_space)
        self.encoders = [self.encoder]
        self.core = HomeostaticCore(cfg, self.encoder.get_out_size())
        self.decoder = nn.Identity()
        decoder_out_size = self.core.get_out_size()
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)
        self.critic_food = nn.Linear(decoder_out_size, 1)
        self.critic_drink = nn.Linear(decoder_out_size, 1)
        self.critic_energy = nn.Linear(decoder_out_size, 1)
        self.critic_health = nn.Linear(decoder_out_size, 1)
        self.icm = IntrinsicCuriosityModule(cfg, obs_space, action_space)

    def forward(self, normalized_obs_dict, rnn_states, values_only=False, action_mask=None):
        head_output = self.encoder(normalized_obs_dict)
        core_output, new_rnn_states = self.core(head_output, rnn_states)
        decoder_output = self.decoder(core_output)
        
        result = {
            "value_food": self.critic_food(decoder_output).squeeze(-1),
            "value_drink": self.critic_drink(decoder_output).squeeze(-1),
            "value_energy": self.critic_energy(decoder_output).squeeze(-1),
            "value_health": self.critic_health(decoder_output).squeeze(-1),
            "new_rnn_states": new_rnn_states,
        }
        
        if values_only:
            return result
        
        # This call returns a distribution object, but the API requires raw tensors.
        action_distribution = self.action_parameterization(decoder_output)
        
        # Extract the logits tensor from the distribution object.
        # This is the expected output for discrete action spaces.
        result["action_logits"] = action_distribution.logits
            
        return result