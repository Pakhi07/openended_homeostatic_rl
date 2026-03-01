"""Training entrypoint for modular PPO + homeostatic critics + curiosity."""

from __future__ import annotations

from typing import Optional
import sys

import gymnasium
import homeostatic_crafter  # noqa: F401

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from hcrafter_params import hcrafter_override_defaults
from modular_homeostatic_model import ModularHomeostaticActorCritic
from recorder import EpisodeRecorder, StatsRecorder, VideoRecorder


def make_custom_env(
    full_env_name: str = "HomeostaticCrafter-v1",
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del cfg, env_config, render_mode
    return gymnasium.make(full_env_name, size=(64, 64), random_internal=True)


def make_custom_env_enjoy(
    full_env_name: str = "HomeostaticCrafter-v1",
    cfg=None,
    env_config=None,
    render_mode: Optional[str] = None,
):
    del cfg, env_config, render_mode
    env = gymnasium.make(
        full_env_name,
        size=(64, 64),
        random_internal=False,
        render_mode="human",
    )

    from datetime import datetime

    run_name = "enjoy_modular_icm_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env = StatsRecorder(env, run_name)
    env = VideoRecorder(env, run_name, (512, 512))
    env = EpisodeRecorder(env, run_name)
    return env


def register_custom_env_envs(enjoy: bool = False):
    if enjoy:
        register_env("HomeostaticCrafter-v1", make_custom_env_enjoy)
    else:
        register_env("HomeostaticCrafter-v1", make_custom_env)


def register_modular_homeostatic_models():
    global_model_factory().register_actor_critic_factory(ModularHomeostaticActorCritic)


def add_modular_homeostatic_args(parser):
    parser.add_argument(
        "--homeo_setpoints",
        default="9.0,9.0,9.0,9.0",
        type=str,
        help="Comma-separated setpoints for [health,food,drink,energy]",
    )
    parser.add_argument(
        "--homeo_weights",
        default="1.0,1.0,1.0,1.0",
        type=str,
        help="Comma-separated drive weights for [health,food,drink,energy]",
    )
    parser.add_argument(
        "--homeo_reward_scale",
        default=1.0,
        type=float,
        help="Scale for homeostatic reward stream",
    )
    parser.add_argument(
        "--curiosity_reward_scale",
        default=0.01,
        type=float,
        help="Scale for curiosity intrinsic reward stream",
    )
    parser.add_argument(
        "--icm_beta",
        default=0.2,
        type=float,
        help="Forward-vs-inverse curiosity loss mixing coefficient",
    )
    parser.add_argument(
        "--icm_loss_coeff",
        default=1.0,
        type=float,
        help="Scale for curiosity loss in total objective",
    )
    parser.add_argument(
        "--gate_entropy_coeff",
        default=0.01,
        type=float,
        help="Entropy regularization coefficient for gate weights",
    )
    parser.add_argument(
        "--gate_balance_coeff",
        default=0.01,
        type=float,
        help="KL-to-uniform gate balancing coefficient",
    )
    parser.add_argument(
        "--shared_latent_dim",
        default=512,
        type=int,
        help="Latent size after shared encoder",
    )
    parser.add_argument(
        "--specialist_latent_dim",
        default=256,
        type=int,
        help="Latent size used by each specialist torso",
    )
    parser.add_argument(
        "--icm_feature_dim",
        default=288,
        type=int,
        help="Feature dimension for curiosity forward/inverse models",
    )


def parse_args(argv=None, evaluation: bool = False):
    parser, _ = parse_sf_args(argv=argv, evaluation=evaluation)
    hcrafter_override_defaults(parser)
    add_modular_homeostatic_args(parser)
    cfg = parse_full_cfg(parser, argv)
    cfg.learner_custom_class = "modular_homeostatic_learner.ModularHomeostaticLearner"
    cfg.algo = "PPO"
    return cfg


def main():
    register_custom_env_envs()
    register_modular_homeostatic_models()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
