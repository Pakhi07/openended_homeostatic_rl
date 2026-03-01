"""Custom PPO learner for modular homeostatic critics with curiosity."""

from __future__ import annotations

from typing import Any, Dict, Iterable, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from modular_homeostatic_model import DRIVE_ORDER

try:
    from sample_factory.algo.learning.learner_worker import Learner
    from sample_factory.utils.typing import Config
except ModuleNotFoundError:  # pragma: no cover - fallback for local tests without Sample Factory
    Config = Any

    class Learner:
        def __init__(self, cfg: Config, policy_versions_root=None, policy_version=None):
            self.cfg = cfg
            self.gamma = float(getattr(cfg, "gamma", 0.99))
            self.gae_lambda = float(getattr(cfg, "gae_lambda", 0.95))
            self.value_loss_coeff = float(getattr(cfg, "value_loss_coeff", 0.5))
            self.entropy_coeff = float(getattr(cfg, "exploration_loss_coeff", 0.0))
            self.actor_critic = None


class PPO_Losses(NamedTuple):
    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy_loss: Tensor
    kl_loss: Tensor
    kl_div: Tensor


def parse_float_list(spec: Any, expected_len: int) -> Sequence[float]:
    """Parse comma separated floats from config string/list."""
    if isinstance(spec, str):
        values = [float(token.strip()) for token in spec.split(",") if token.strip()]
    elif isinstance(spec, Iterable):
        values = [float(v) for v in spec]
    else:
        raise TypeError(f"Unsupported float list specification type: {type(spec)}")

    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)} from {spec}")
    return values


def compute_drive(measurements: Tensor, setpoints: Tensor, weights: Tensor) -> Tensor:
    """Compute weighted squared homeostatic drive for each stream."""
    if measurements.shape[-1] < len(DRIVE_ORDER):
        raise ValueError(
            f"measurements last dim must be >= {len(DRIVE_ORDER)}, got {measurements.shape[-1]}"
        )
    deltas = measurements[..., : len(DRIVE_ORDER)] - setpoints
    return weights * (deltas ** 2)


def compute_homeostatic_rewards(
    measurements_t: Tensor,
    measurements_tp1: Tensor,
    setpoints: Tensor,
    weights: Tensor,
) -> Dict[str, Tensor]:
    """Compute reward as decrease in drive for each homeostatic stream."""
    drive_t = compute_drive(measurements_t, setpoints, weights)
    drive_tp1 = compute_drive(measurements_tp1, setpoints, weights)
    reward_tensor = drive_t - drive_tp1
    return {name: reward_tensor[:, idx] for idx, name in enumerate(DRIVE_ORDER)}


def _reshape_time_batch(flat_tensor: Tensor, recurrence: int) -> Tuple[Tensor, int, int]:
    """Reshape [B] -> [T, N] using recurrence."""
    batch_items = flat_tensor.shape[0]
    if recurrence <= 0 or batch_items % recurrence != 0:
        recurrence = 1
    num_envs = batch_items // recurrence
    time_env = flat_tensor.view(num_envs, recurrence).transpose(0, 1).contiguous()
    return time_env, recurrence, num_envs


def _flatten_time_batch(time_env_tensor: Tensor) -> Tensor:
    """Flatten [T, N] -> [B]."""
    return time_env_tensor.transpose(0, 1).reshape(-1)


def gae_advantages_from_values(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """GAE over [T, N] tensors using shifted values for bootstrap."""
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(rewards.shape[0])):
        if t + 1 < rewards.shape[0]:
            next_value = values[t + 1]
        else:
            next_value = torch.zeros_like(values[t])

        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
        advantages[t] = last_advantage

    value_targets = advantages + values
    return advantages, value_targets


class ModularHomeostaticLearner(Learner):
    """Learner that combines modular PPO, homeostatic reward, and curiosity."""

    def __init__(self, cfg: Config, policy_versions_root, policy_version, ppo_clip_value):
        super().__init__(cfg, policy_versions_root, policy_version)
        self.ppo_clip_value = float(ppo_clip_value)
        self.recurrence = int(getattr(cfg, "recurrence", getattr(cfg, "rollout", 1)))

        self.homeo_reward_scale = float(getattr(cfg, "homeo_reward_scale", 1.0))
        self.curiosity_reward_scale = float(getattr(cfg, "curiosity_reward_scale", 0.01))
        self.icm_beta = float(getattr(cfg, "icm_beta", 0.2))
        self.icm_loss_coeff = float(getattr(cfg, "icm_loss_coeff", 1.0))
        self.gate_entropy_coeff = float(getattr(cfg, "gate_entropy_coeff", 0.01))
        self.gate_balance_coeff = float(getattr(cfg, "gate_balance_coeff", 0.01))

        self._setpoint_values = parse_float_list(
            getattr(cfg, "homeo_setpoints", "9.0,9.0,9.0,9.0"),
            expected_len=len(DRIVE_ORDER),
        )
        self._weight_values = parse_float_list(
            getattr(cfg, "homeo_weights", "1.0,1.0,1.0,1.0"),
            expected_len=len(DRIVE_ORDER),
        )

    def _homeo_constants(self, device: torch.device, dtype: torch.dtype) -> Tuple[Tensor, Tensor]:
        setpoints = torch.tensor(self._setpoint_values, device=device, dtype=dtype)
        weights = torch.tensor(self._weight_values, device=device, dtype=dtype)
        return setpoints, weights

    def _compute_value_targets(
        self, rewards: Tensor, dones: Tensor, values: Tensor
    ) -> Tuple[Tensor, Tensor]:
        rewards_tn, _, _ = _reshape_time_batch(rewards, self.recurrence)
        dones_tn, _, _ = _reshape_time_batch(dones.float(), self.recurrence)
        values_tn, _, _ = _reshape_time_batch(values, self.recurrence)
        advantages_tn, targets_tn = gae_advantages_from_values(
            rewards_tn,
            dones_tn,
            values_tn,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        return _flatten_time_batch(advantages_tn), _flatten_time_batch(targets_tn)

    def _build_loss(self, mb, policy_outputs):
        action_dist, model_outputs, _ = policy_outputs

        values_dict = {
            "health": model_outputs["value_health"].view(-1),
            "food": model_outputs["value_food"].view(-1),
            "drink": model_outputs["value_drink"].view(-1),
            "energy": model_outputs["value_energy"].view(-1),
        }
        gate_weights = model_outputs["gate_weights"]

        actions = mb.actions
        if actions.ndim > 1:
            actions_flat = actions.squeeze(-1)
        else:
            actions_flat = actions
        actions_long = actions_flat.long()

        log_prob_actions = action_dist.log_prob(actions)
        if log_prob_actions.ndim > 1:
            log_prob_actions = log_prob_actions.sum(dim=-1)
        old_log_prob_actions = mb.log_prob_actions
        if old_log_prob_actions.ndim > 1:
            old_log_prob_actions = old_log_prob_actions.squeeze(-1)

        phi_next_target, pred_phi_next, pred_action_logits = self.actor_critic.icm(
            mb.obs, mb.next_obs, actions_long
        )
        forward_loss_per_step = F.mse_loss(
            pred_phi_next, phi_next_target, reduction="none"
        ).mean(dim=-1)
        inverse_loss_per_step = F.cross_entropy(
            pred_action_logits, actions_long, reduction="none"
        )
        intrinsic_reward = self.curiosity_reward_scale * (0.5 * forward_loss_per_step).detach()

        measurements_t = mb.obs["measurements"].float()
        measurements_tp1 = mb.next_obs["measurements"].float()
        setpoints, weights = self._homeo_constants(
            device=measurements_t.device, dtype=measurements_t.dtype
        )
        homeo_rewards = compute_homeostatic_rewards(
            measurements_t, measurements_tp1, setpoints=setpoints, weights=weights
        )

        advantages = {}
        value_targets = {}
        for stream in DRIVE_ORDER:
            stream_reward = self.homeo_reward_scale * homeo_rewards[stream] + intrinsic_reward
            stream_adv, stream_targets = self._compute_value_targets(
                rewards=stream_reward,
                dones=mb.dones.float(),
                values=values_dict[stream],
            )
            advantages[stream] = stream_adv
            value_targets[stream] = stream_targets

        stacked_advantages = torch.stack([advantages[name] for name in DRIVE_ORDER], dim=1)
        policy_advantages = torch.sum(gate_weights.detach() * stacked_advantages, dim=1)
        if getattr(self.cfg, "normalize_advantages", True):
            policy_advantages = (policy_advantages - policy_advantages.mean()) / (
                policy_advantages.std() + 1e-8
            )

        ratio = torch.exp(log_prob_actions - old_log_prob_actions)
        unclipped = ratio * policy_advantages
        clipped = (
            torch.clamp(ratio, 1.0 - self.ppo_clip_value, 1.0 + self.ppo_clip_value)
            * policy_advantages
        )
        policy_loss = -torch.min(unclipped, clipped).mean()

        total_value_loss = torch.tensor(0.0, device=policy_loss.device)
        for stream in DRIVE_ORDER:
            total_value_loss = total_value_loss + F.mse_loss(
                values_dict[stream], value_targets[stream]
            )

        entropy = action_dist.entropy()
        if entropy.ndim > 1:
            entropy = entropy.sum(dim=-1)
        entropy_loss = -self.entropy_coeff * entropy.mean()

        forward_loss = forward_loss_per_step.mean()
        inverse_loss = inverse_loss_per_step.mean()
        icm_loss = (1.0 - self.icm_beta) * inverse_loss + self.icm_beta * forward_loss

        gate_entropy = -(gate_weights * torch.log(gate_weights.clamp_min(1e-8))).sum(dim=1).mean()
        gate_entropy_loss = -gate_entropy
        mean_gate_weights = gate_weights.mean(dim=0)
        uniform = torch.full_like(mean_gate_weights, 1.0 / len(DRIVE_ORDER))
        gate_balance_kl = torch.sum(
            mean_gate_weights * torch.log(mean_gate_weights.clamp_min(1e-8) / uniform)
        )

        total_loss = (
            policy_loss
            + self.value_loss_coeff * total_value_loss
            + entropy_loss
            + self.icm_loss_coeff * icm_loss
            + self.gate_entropy_coeff * gate_entropy_loss
            + self.gate_balance_coeff * gate_balance_kl
        )

        zero = torch.tensor(0.0, device=total_loss.device)
        return PPO_Losses(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=total_value_loss,
            entropy_loss=entropy_loss,
            kl_loss=zero,
            kl_div=zero,
        )
