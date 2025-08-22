import torch
import torch.nn as nn
from torch.distributions import Categorical

from sample_factory.model.actor_critic import ActorCritic
from hcrafter_model import make_hcrafter_encoder

class ModularActorCritic(ActorCritic):
    def __init__(self, cfg, obs_space, action_space):
        nn.Module.__init__(self)

        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actions = action_space.n
        self.num_agents = 1

        self.num_selves = cfg.num_selves

        self.encoder = make_hcrafter_encoder(cfg, obs_space)
        self.encoders = [self.encoder]
        encoder_out_dim = self.encoder.get_out_size()

        self.critic_head = nn.Linear(encoder_out_dim, 1)

        self.manager_net = nn.Linear(encoder_out_dim, self.num_selves)
        self.selves_nets = nn.ModuleList(
            [nn.Linear(encoder_out_dim, self.action_space.n) for _ in range(self.num_selves)]
        )

        self.apply(self.initialize_weights)
        self.train()

    # --- ADD THIS METHOD ---
    def normalize_obs(self, obs):
        # We override this method to bypass the default ObservationNormalizer,
        # which is not created in our custom __init__.
        # Our custom encoder handles any necessary normalization.
        return obs

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
        final_action_logits = torch.log(mixed_action_probs + 1e-8)

        # --- FIX: Sample an action and add it to the output dictionary ---
        # Create a distribution from the mixed probabilities
        action_distribution = Categorical(probs=mixed_action_probs)
        # Sample the action
        actions = action_distribution.sample()

        # The forward pass in sample-factory must return a dictionary
        result = dict(
            action_logits=final_action_logits,
            values=value,
            rnn_states=rnn_states,  # Pass rnn_states through, even if unused
            actions=actions, # Add the sampled actions to the dictionary
        )
        return result