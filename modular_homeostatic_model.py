"""Modular actor-critic model with specialist critics and curiosity module."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sample_factory.model.actor_critic import ActorCritic
    from sample_factory.model.encoder import Encoder
    from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
except ModuleNotFoundError:  # pragma: no cover - fallback for local tests without Sample Factory
    Config = Any
    ObsSpace = Dict[str, Any]
    ActionSpace = Any

    class Encoder(nn.Module):
        def __init__(self, cfg: Config):
            super().__init__()
            self.cfg = cfg

        def get_out_size(self) -> int:
            raise NotImplementedError

    class _FallbackActionHead(nn.Module):
        def __init__(self, input_size: int, num_actions: int):
            super().__init__()
            self.linear = nn.Linear(input_size, num_actions)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    class ActorCritic(nn.Module):
        def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
            super().__init__()
            self.obs_space = obs_space
            self.action_space = action_space
            self.cfg = cfg

        def get_action_parameterization(self, input_size: int):
            num_actions = int(getattr(self.action_space, "n", 1))
            return _FallbackActionHead(input_size, num_actions)


DRIVE_ORDER: Sequence[str] = ("health", "food", "drink", "energy")


def _maybe_measurement_dim(obs_space: ObsSpace) -> int:
    if hasattr(obs_space, "keys") and "measurements" in obs_space.keys():
        return int(obs_space["measurements"].shape[0])
    return 0


class SharedHomeostaticEncoder(Encoder):
    """Shared encoder for image observations and optional measurements."""

    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        image_channels = int(obs_space["obs"].shape[0])
        self.measurement_dim = _maybe_measurement_dim(obs_space)

        self.image_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_space["obs"].shape)
            image_latent_dim = int(self.image_encoder(dummy).shape[1])

        self.measurement_encoder: Optional[nn.Module] = None
        measurement_latent_dim = 0
        if self.measurement_dim > 0:
            self.measurement_encoder = nn.Sequential(
                nn.Linear(self.measurement_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )
            measurement_latent_dim = 64

        shared_dim = int(getattr(cfg, "shared_latent_dim", 512))
        self.shared_projection = nn.Sequential(
            nn.Linear(image_latent_dim + measurement_latent_dim, shared_dim),
            nn.ReLU(),
        )
        self._out_size = shared_dim

    def get_out_size(self) -> int:
        return self._out_size

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs_dict["obs"].float() / 255.0
        image_latent = self.image_encoder(image)
        if self.measurement_encoder is None:
            return self.shared_projection(image_latent)

        measurements = obs_dict["measurements"].float()
        measurement_latent = self.measurement_encoder(measurements)
        return self.shared_projection(torch.cat([image_latent, measurement_latent], dim=1))


class CuriosityFeatureEncoder(nn.Module):
    """Feature encoder phi(s) used by NoReward-RL style curiosity."""

    def __init__(self, obs_space: ObsSpace, feature_dim: int):
        super().__init__()
        image_channels = int(obs_space["obs"].shape[0])
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *obs_space["obs"].shape)
            conv_out_dim = int(self.encoder(dummy).shape[1])
        self.proj = nn.Linear(conv_out_dim, feature_dim)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        image = obs_dict["obs"].float() / 255.0
        return self.proj(self.encoder(image))


class IntrinsicCuriosityModule(nn.Module):
    """Forward/inverse curiosity module used for intrinsic rewards."""

    def __init__(self, obs_space: ObsSpace, action_space: ActionSpace, feature_dim: int = 288):
        super().__init__()
        self.action_dim = int(action_space.n)
        self.feature_encoder = CuriosityFeatureEncoder(obs_space, feature_dim)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    @staticmethod
    def intrinsic_reward_from_prediction(
        predicted_phi_next: torch.Tensor,
        target_phi_next: torch.Tensor,
    ) -> torch.Tensor:
        sq_error = (predicted_phi_next - target_phi_next) ** 2
        return 0.5 * sq_error.mean(dim=-1)

    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ):
        phi_state = self.feature_encoder(obs_dict)
        phi_next = self.feature_encoder(next_obs_dict)

        if action.ndim > 1:
            action = action.squeeze(-1)
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        pred_phi_next = self.forward_model(torch.cat([phi_state, action_one_hot], dim=1))
        pred_action_logits = self.inverse_model(torch.cat([phi_state, phi_next], dim=1))
        return phi_next.detach(), pred_phi_next, pred_action_logits

    def get_intrinsic_reward(
        self,
        obs_dict: Dict[str, torch.Tensor],
        next_obs_dict: Dict[str, torch.Tensor],
        action: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            phi_next_target, pred_phi_next, _ = self.forward(obs_dict, next_obs_dict, action)
            return self.intrinsic_reward_from_prediction(pred_phi_next, phi_next_target)


class ModularHomeostaticActorCritic(ActorCritic):
    """Actor-critic with specialist critics and soft hierarchical gating."""

    def __init__(self, cfg: Config, obs_space: ObsSpace, action_space: ActionSpace):
        super().__init__(obs_space, action_space, cfg)
        self.obs_space = obs_space

        self.encoder = SharedHomeostaticEncoder(cfg, obs_space)
        self.encoders = [self.encoder]
        shared_dim = self.encoder.get_out_size()
        specialist_dim = int(getattr(cfg, "specialist_latent_dim", 256))

        self.specialist_torso_health = nn.Sequential(
            nn.Linear(shared_dim, specialist_dim),
            nn.ReLU(),
            nn.Linear(specialist_dim, specialist_dim),
            nn.ReLU(),
        )
        self.specialist_torso_food = nn.Sequential(
            nn.Linear(shared_dim, specialist_dim),
            nn.ReLU(),
            nn.Linear(specialist_dim, specialist_dim),
            nn.ReLU(),
        )
        self.specialist_torso_drink = nn.Sequential(
            nn.Linear(shared_dim, specialist_dim),
            nn.ReLU(),
            nn.Linear(specialist_dim, specialist_dim),
            nn.ReLU(),
        )
        self.specialist_torso_energy = nn.Sequential(
            nn.Linear(shared_dim, specialist_dim),
            nn.ReLU(),
            nn.Linear(specialist_dim, specialist_dim),
            nn.ReLU(),
        )

        gate_input_dim = shared_dim + self.encoder.measurement_dim
        self.gating_network = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, len(DRIVE_ORDER)),
        )

        self.critic_health = nn.Linear(specialist_dim, 1)
        self.critic_food = nn.Linear(specialist_dim, 1)
        self.critic_drink = nn.Linear(specialist_dim, 1)
        self.critic_energy = nn.Linear(specialist_dim, 1)

        self.decoder = nn.Identity()
        self.action_parameterization = self.get_action_parameterization(specialist_dim)
        self.icm = IntrinsicCuriosityModule(
            obs_space=obs_space,
            action_space=action_space,
            feature_dim=int(getattr(cfg, "icm_feature_dim", 288)),
        )

    @staticmethod
    def _extract_action_logits(action_output: Any) -> torch.Tensor:
        if torch.is_tensor(action_output):
            return action_output

        if isinstance(action_output, tuple):
            for item in action_output:
                if torch.is_tensor(item):
                    return item
                if hasattr(item, "logits") and torch.is_tensor(item.logits):
                    return item.logits

        if hasattr(action_output, "logits") and torch.is_tensor(action_output.logits):
            return action_output.logits

        if hasattr(action_output, "probs") and torch.is_tensor(action_output.probs):
            return torch.log(action_output.probs.clamp_min(1e-8))

        raise TypeError(
            f"Unsupported action parameterization output type: {type(action_output)}"
        )

    def _build_specialist_latents(self, shared_latent: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                self.specialist_torso_health(shared_latent),
                self.specialist_torso_food(shared_latent),
                self.specialist_torso_drink(shared_latent),
                self.specialist_torso_energy(shared_latent),
            ],
            dim=1,
        )

    def forward(
        self,
        normalized_obs_dict: Dict[str, torch.Tensor],
        rnn_states: torch.Tensor,
        values_only: bool = False,
        action_mask: Optional[torch.Tensor] = None,
    ):
        del action_mask  # action masking is not used in this environment
        shared_latent = self.encoder(normalized_obs_dict)
        specialist_latents = self._build_specialist_latents(shared_latent)

        measurements = normalized_obs_dict.get("measurements", None)
        if measurements is not None:
            gate_input = torch.cat([shared_latent, measurements.float()], dim=1)
        else:
            gate_input = shared_latent

        gate_logits = self.gating_network(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        gated_latent = torch.sum(specialist_latents * gate_weights.unsqueeze(-1), dim=1)
        decoder_output = self.decoder(gated_latent)

        result = {
            "value_health": self.critic_health(specialist_latents[:, 0]).squeeze(-1),
            "value_food": self.critic_food(specialist_latents[:, 1]).squeeze(-1),
            "value_drink": self.critic_drink(specialist_latents[:, 2]).squeeze(-1),
            "value_energy": self.critic_energy(specialist_latents[:, 3]).squeeze(-1),
            "gate_logits": gate_logits,
            "gate_weights": gate_weights,
            "new_rnn_states": rnn_states,
        }
        if values_only:
            return result

        action_output = self.action_parameterization(decoder_output)
        result["action_logits"] = self._extract_action_logits(action_output)
        return result


def make_modular_homeostatic_actor_critic(
    cfg: Config,
    obs_space: ObsSpace,
    action_space: ActionSpace,
) -> ModularHomeostaticActorCritic:
    """Factory helper for Sample Factory registration."""
    return ModularHomeostaticActorCritic(cfg, obs_space, action_space)
