from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import modular_homeostatic_learner as learner_mod
from modular_homeostatic_learner import (
    ModularHomeostaticLearner,
    compute_homeostatic_rewards,
    parse_float_list,
)


def test_parse_float_list():
    parsed = parse_float_list("9.0, 8.5,7,6.25", expected_len=4)
    assert parsed == [9.0, 8.5, 7.0, 6.25]


def test_homeostatic_reward_matches_manual_calculation():
    measurements_t = torch.tensor(
        [
            [10.0, 8.0, 9.0, 7.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    measurements_tp1 = torch.tensor(
        [
            [9.5, 8.5, 8.0, 8.0],
            [10.0, 9.0, 8.0, 9.0],
        ]
    )
    setpoints = torch.tensor([9.0, 9.0, 9.0, 9.0])
    weights = torch.tensor([1.0, 2.0, 3.0, 4.0])

    rewards = compute_homeostatic_rewards(measurements_t, measurements_tp1, setpoints, weights)

    # health row0: 1*(10-9)^2 - 1*(9.5-9)^2 = 1 - 0.25 = 0.75
    assert torch.allclose(rewards["health"][0], torch.tensor(0.75))
    # drink row0: 3*(9-9)^2 - 3*(8-9)^2 = 0 - 3 = -3
    assert torch.allclose(rewards["drink"][0], torch.tensor(-3.0))
    # row1 food remains at setpoint in both states
    assert torch.allclose(rewards["food"][1], torch.tensor(0.0))


@pytest.mark.skipif(
    learner_mod.Learner.__module__.startswith("sample_factory"),
    reason="Standalone loss assembly test uses fallback learner in environments without Sample Factory.",
)
def test_loss_assembly_returns_finite_scalars():
    cfg = SimpleNamespace(
        gamma=0.99,
        gae_lambda=0.95,
        value_loss_coeff=0.5,
        exploration_loss_coeff=0.01,
        normalize_advantages=True,
        recurrence=4,
        rollout=4,
        homeo_setpoints="9.0,9.0,9.0,9.0",
        homeo_weights="1.0,1.0,1.0,1.0",
        homeo_reward_scale=1.0,
        curiosity_reward_scale=0.01,
        icm_beta=0.2,
        icm_loss_coeff=1.0,
        gate_entropy_coeff=0.01,
        gate_balance_coeff=0.01,
    )
    learner = ModularHomeostaticLearner(cfg, policy_versions_root=None, policy_version=None, ppo_clip_value=0.2)

    batch_size = 8
    action_dim = 17

    class MockActorCritic:
        def icm(self, obs, next_obs, actions):
            del obs, next_obs, actions
            phi_next_target = torch.randn(batch_size, 16)
            pred_phi_next = phi_next_target + 0.1 * torch.randn(batch_size, 16)
            pred_action_logits = torch.randn(batch_size, action_dim)
            return phi_next_target, pred_phi_next, pred_action_logits

    class MockActionDist:
        def log_prob(self, actions):
            del actions
            return torch.zeros(batch_size)

        def entropy(self):
            return torch.ones(batch_size)

    learner.actor_critic = MockActorCritic()

    mb = SimpleNamespace(
        actions=torch.randint(0, action_dim, (batch_size, 1)),
        log_prob_actions=torch.zeros(batch_size),
        dones=torch.zeros(batch_size),
        obs={
            "obs": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
            "measurements": torch.randn(batch_size, 4),
        },
        next_obs={
            "obs": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
            "measurements": torch.randn(batch_size, 4),
        },
    )
    model_outputs = {
        "value_health": torch.randn(batch_size),
        "value_food": torch.randn(batch_size),
        "value_drink": torch.randn(batch_size),
        "value_energy": torch.randn(batch_size),
        "gate_weights": torch.softmax(torch.randn(batch_size, 4), dim=-1),
    }

    losses = learner._build_loss(mb, (MockActionDist(), model_outputs, None))
    assert torch.isfinite(losses.total_loss)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.value_loss)
    assert torch.isfinite(losses.entropy_loss)
