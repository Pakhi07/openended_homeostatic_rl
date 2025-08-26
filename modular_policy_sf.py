
# File: modular_policy_sf.py (Fixed to return Sample Factory expected keys)
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium.spaces as spaces

from sample_factory.model.actor_critic import ActorCritic
from sample_factory.model.model_utils import nonlinearity
from hcrafter_model import make_hcrafter_encoder


class ModuleHead(nn.Module):
    """
    A simple MLP head that produces policy logits and a scalar value.
    """
    def __init__(self, cfg, core_output_size: int, num_actions: int, mlp_layers):
        super().__init__()
        layers = []
        in_size = core_output_size
        for layer_size in mlp_layers:
            layers.append(nn.Linear(in_size, layer_size))
            layers.append(nonlinearity(cfg))
            in_size = layer_size
        self.decoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.decoder_out_size = in_size if layers else core_output_size
        self.policy_linear = nn.Linear(self.decoder_out_size, num_actions)
        self.value_linear = nn.Linear(self.decoder_out_size, 1)

    def forward(self, core_output: torch.Tensor):
        x = self.decoder(core_output)
        policy_logits = self.policy_linear(x)
        value = self.value_linear(x).squeeze(-1)  # [B]
        return policy_logits, value


class ModularActorCritic(ActorCritic):
    def _make_new_rnn_states(self, batch_size, device, dtype, rnn_states):
        """Return a tensor for new_rnn_states compatible with SF expectations.
        If no RNN is used, we provide a zero-sized (or small zero) tensor so .float() works.
        """
        if isinstance(rnn_states, torch.Tensor):
            return rnn_states
        if rnn_states is not None:
            try:
                return torch.as_tensor(rnn_states, device=device, dtype=dtype)
            except Exception:
                pass
        # Fallback: build zeros with plausible shape
        rnn_size = getattr(self, 'rnn_size', getattr(self.cfg, 'rnn_size', 0))
        num_layers = getattr(self, 'rnn_num_layers', getattr(self.cfg, 'rnn_num_layers', 1))
        if rnn_size <= 0:
            # zero-sized last dim still works for .float(); ensure 3D shape [L, B, 0]
            return torch.zeros((num_layers, batch_size, 0), device=device, dtype=dtype)
        return torch.zeros((num_layers, batch_size, rnn_size), device=device, dtype=dtype)

    """
    Modular policy for Homeostatic Crafter that aggregates multiple heads
    ("selves") by averaging their logits and values.

    IMPORTANT: Returns the keys Sample Factory expects:
        - actions               [B] int64
        - log_prob_actions      [B] float32
        - values                [B] float32
        - policy_logits         [B, A] float32
        - rnn_states            (passed through)
    """
    def __init__(self, cfg, obs_space, action_space):
        super().__init__(obs_space, action_space, cfg)

        if isinstance(action_space, spaces.Box):
            raise NotImplementedError("Continuous actions not supported in this modular policy.")

        # Encoder (built on CPU; Sample Factory will move to the proper device)
        self.encoder = make_hcrafter_encoder(cfg, obs_space)

        # Ensure compatibility with Sample Factory base class expectations
        # which refer to self.encoders[0] in multiple utility paths.
        self.encoders = nn.ModuleList([self.encoder])

        # Number of module heads ("selves")
        self.num_selves = getattr(cfg, "num_selves", 1)
        head_mlp_layers = getattr(cfg, "head_mlp_layers", [256])

        # Module heads
        core_out_size = self.encoder.get_out_size()
        self.module_heads = nn.ModuleList([
            ModuleHead(cfg, core_out_size, action_space.n, head_mlp_layers)
            for _ in range(self.num_selves)
        ])

    def forward(self, obs_dict, rnn_states=None, **kwargs):
        """
        Returns a full dict compatible with Sample Factory's inference_worker.
        Shapes (batch = B, actions = A):
          - policy_logits: [B, A]
          - values:        [B]
          - actions:       [B]
          - log_prob_actions: [B]
        """
        # Encode observations -> core features
        core_output = self.encoder(obs_dict)  # [B, F]

        # Run all heads
        logits_list = []
        values_list = []
        for head in self.module_heads:
            policy_logits_i, value_i = head(core_output)
            logits_list.append(policy_logits_i)
            values_list.append(value_i)

        # Aggregate across heads by simple mean
        stacked_logits = torch.stack(logits_list, dim=0)      # [H, B, A]
        stacked_values = torch.stack(values_list, dim=0)      # [H, B]
        policy_logits = stacked_logits.mean(dim=0)            # [B, A]
        values = stacked_values.mean(dim=0)                   # [B]

        # Sample actions + compute log-probabilities
        dist = Categorical(logits=policy_logits)
        actions = dist.sample()                                # [B]
        log_prob_actions = dist.log_prob(actions)              # [B]

        # Build new_rnn_states tensor (safe even if no RNN in use)
        batch_size = policy_logits.size(0)
        new_rnn_states = self._make_new_rnn_states(
            batch_size, policy_logits.device, policy_logits.dtype, rnn_states
        )

        # Prepare output dict with expected keys
        result = {
            "policy_logits": policy_logits.float(),
            "action_logits": policy_logits.float(),   # alias for SF versions that expect this
            "values": values.float(),
            "actions": actions.long(),
            "log_prob_actions": log_prob_actions.float(),
            "rnn_states": new_rnn_states,
            "new_rnn_states": new_rnn_states,
            # Extra introspection (optional)
            "module_logits": stacked_logits.float(),
            "module_values": stacked_values.float(),
        }
        print({
            "policy_logits": policy_logits.shape,
            "values": values.shape,
            "actions": actions.shape,
            "log_prob_actions": log_prob_actions.shape,
            "new_rnn_states": tuple(new_rnn_states.shape) if isinstance(new_rnn_states, torch.Tensor) else type(new_rnn_states),
        })
        return result
