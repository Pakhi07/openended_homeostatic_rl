from typing import Optional
import argparse
import sys

import gymnasium
import homeostatic_crafter

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

from hcrafter_model import make_hcrafter_encoder
from hcrafter_params import hcrafter_override_defaults
from recorder import StatsRecorder, VideoRecorder, EpisodeRecorder

# --- CHANGE 1: Import your new modular model ---
from modular_policy_sf import ModularActorCritic


def make_custom_env(
        full_env_name: str = 'HomeostaticCrafter-v1',
        cfg=None,
        env_config=None,
        render_mode: Optional[str] = None,
):
    env = gymnasium.make(full_env_name, size=(64, 64), random_internal=True)
    return env


def make_custom_env_enjoy(
        full_env_name: str = 'HomeostaticCrafter-v1',
        cfg=None,
        env_config=None,
        render_mode: Optional[str] = None,
):
    env = gymnasium.make(full_env_name, size=(64, 64),
                         random_internal=False,  # False by default!
                         render_mode="human")
    
    from datetime import datetime
    run_name = "enjoy_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env = StatsRecorder(env, run_name)
    env = VideoRecorder(env, run_name, (512, 512))
    env = EpisodeRecorder(env, run_name)
    return env


def register_custom_env_envs(enjoy=False):
    if enjoy:
        register_env('HomeostaticCrafter-v1', make_custom_env_enjoy)
    else:
        register_env('HomeostaticCrafter-v1', make_custom_env)


def register_hcrafter_models():
    global_model_factory().register_encoder_factory(make_hcrafter_encoder)
    # --- CHANGE 2: Register your custom ActorCritic model with sample-factory ---
    global_model_factory().register_actor_critic_factory(ModularActorCritic)


def parse_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    
    hcrafter_override_defaults(parser)

    # --- CHANGE 3: Add an argument to control the number of "selves" ---
    parser.add_argument('--num_selves', default=4, type=int, help='Number of expert selves in the modular policy')
    
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    register_custom_env_envs()
    register_hcrafter_models()
    
    cfg = parse_args()
    
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())