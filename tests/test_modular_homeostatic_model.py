from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from modular_homeostatic_model import ModularHomeostaticActorCritic


class DummySpace:
    def __init__(self, shape):
        self.shape = shape


@pytest.fixture
def dummy_inputs():
    obs_space = {
        "obs": DummySpace((3, 64, 64)),
        "measurements": DummySpace((4,)),
    }
    action_space = SimpleNamespace(n=17)
    cfg = SimpleNamespace(
        shared_latent_dim=256,
        specialist_latent_dim=128,
        icm_feature_dim=64,
    )
    batch_size = 8
    obs = {
        "obs": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
        "measurements": torch.randn(batch_size, 4),
    }
    next_obs = {
        "obs": torch.randint(0, 256, (batch_size, 3, 64, 64), dtype=torch.uint8),
        "measurements": torch.randn(batch_size, 4),
    }
    actions = torch.randint(0, action_space.n, (batch_size,), dtype=torch.long)
    return cfg, obs_space, action_space, obs, next_obs, actions


def test_model_forward_shapes(dummy_inputs):
    cfg, obs_space, action_space, obs, _, _ = dummy_inputs
    model = ModularHomeostaticActorCritic(cfg, obs_space, action_space)
    output = model(obs, rnn_states=torch.zeros(obs["obs"].shape[0], 1))

    assert "action_logits" in output
    assert "gate_weights" in output
    assert "gate_logits" in output
    assert "value_health" in output
    assert "value_food" in output
    assert "value_drink" in output
    assert "value_energy" in output

    assert output["action_logits"].shape == (obs["obs"].shape[0], action_space.n)
    assert output["gate_logits"].shape == (obs["obs"].shape[0], 4)
    assert output["gate_weights"].shape == (obs["obs"].shape[0], 4)
    assert output["value_health"].shape == (obs["obs"].shape[0],)
    assert output["value_food"].shape == (obs["obs"].shape[0],)
    assert output["value_drink"].shape == (obs["obs"].shape[0],)
    assert output["value_energy"].shape == (obs["obs"].shape[0],)

    row_sums = output["gate_weights"].sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_curiosity_forward_and_intrinsic_reward(dummy_inputs):
    cfg, obs_space, action_space, obs, next_obs, actions = dummy_inputs
    model = ModularHomeostaticActorCritic(cfg, obs_space, action_space)

    phi_next_target, pred_phi_next, pred_action_logits = model.icm(obs, next_obs, actions)
    assert phi_next_target.shape == pred_phi_next.shape
    assert pred_action_logits.shape == (obs["obs"].shape[0], action_space.n)

    forward_loss = torch.nn.functional.mse_loss(pred_phi_next, phi_next_target)
    inverse_loss = torch.nn.functional.cross_entropy(pred_action_logits, actions)
    intrinsic_reward = model.icm.intrinsic_reward_from_prediction(pred_phi_next, phi_next_target)

    assert torch.isfinite(forward_loss)
    assert torch.isfinite(inverse_loss)
    assert intrinsic_reward.shape == (obs["obs"].shape[0],)
    assert torch.all(intrinsic_reward >= 0.0)
