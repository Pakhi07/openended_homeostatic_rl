import torch
import torch.nn as nn
from torch.distributions import Categorical

from sample_factory.model.actor_critic import ActorCritic
from hcrafter_model import make_hcrafter_encoder  # Assuming CrafterEncoder is in this file

class ModularActorCritic(ActorCritic):
    """
    A modular Actor-Critic model for Sample-Factory.

    It uses a shared encoder to process observations, a standard critic head to predict
    the value, and a modular "Mixture of Experts" actor head to select actions.
    """
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(cfg, obs_space, action_space)

        # The number of expert "selves" in our modular actor head
        self.num_selves = cfg.num_selves

        # The encoder is responsible for processing the raw observations (image + vectors)
        # This assumes your custom CrafterEncoder is defined and handles the Dict observation space.
        self.encoder = CrafterEncoder(cfg, obs_space)
        encoder_out_dim = self.encoder.get_out_size()

        # The critic head remains a standard single network
        self.critic_head = nn.Linear(encoder_out_dim, 1)

        # --- Define the Modular Actor Heads ---
        # We replace the default single actor_head with our modular structure.
        self.manager_net = nn.Linear(encoder_out_dim, self.num_selves)
        self.selves_nets = nn.ModuleList(
            [nn.Linear(encoder_out_dim, self.action_space.n) for _ in range(self.num_selves)]
        )

        # Initialize weights
        self.apply(self.initialize_weights)

    def forward(self, obs_dict, rnn_states, values_only=False):
        """
        Forward pass through the entire modular network.
        """
        # 1. Feature Extraction using the shared encoder
        encoder_out = self.encoder(obs_dict)

        # 2. Value Estimation (Critic Path)
        value = self.critic_head(encoder_out).squeeze(-1)

        if values_only:
            return dict(values=value)

        # 3. Modular Action Logic (Actor Path)
        manager_logits = self.manager_net(encoder_out)
        manager_dist = Categorical(logits=manager_logits)

        all_selves_logits = torch.stack([self_net(encoder_out) for self_net in self.selves_nets], dim=1)

        selves_probs = torch.softmax(all_selves_logits, dim=-1)
        manager_probs = manager_dist.probs
        
        # Mix the policies to get the final action probability distribution
        mixed_action_probs = torch.sum(manager_probs.unsqueeze(-1) * selves_probs, dim=1)
        
        # Sample-factory expects action_logits. We use the log of the probabilities.
        # Add a small epsilon for numerical stability.
        final_action_logits = torch.log(mixed_action_probs + 1e-8)

        # The forward pass in sample-factory must return a dictionary
        result = dict(
            action_logits=final_action_logits,
            values=value,
            rnn_states=rnn_states,  # Pass rnn_states through, even if unused
        )
        return result