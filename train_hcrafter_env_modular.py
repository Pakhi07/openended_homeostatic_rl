# train_hcrafter_env_modular.py

from typing import Optional
import sys

import gymnasium
import homeostatic_crafter

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl

# 1. Import the new modular encoder factory function
from hcrafter_model_modular import make_modular_hcrafter_encoder
from hcrafter_params import hcrafter_override_defaults
from recorder import StatsRecorder, VideoRecorder, EpisodeRecorder


def make_custom_env(
        full_env_name: str = 'HomeostaticCrafter-v1',
        cfg=None,
        env_config=None,
        render_mode: Optional[str] = None,
):
    # This function is now exactly as it was in the original non-modular script.
    # No wrappers are needed.
    env = gymnasium.make(full_env_name, size=(64, 64), random_internal=True)
    return env


def make_custom_env_enjoy(
        full_env_name: str = 'HomeostaticCrafter-v1',
        cfg=None,
        env_config=None,
        render_mode: Optional[str] = None,
):
    env = gymnasium.make(full_env_name, size=(64, 64),
                         random_internal=False,
                         render_mode="human")
    
    from datetime import datetime
    run_name = "enjoy_modular_" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    env = StatsRecorder(env, run_name)
    env = VideoRecorder(env, run_name, (512, 512))
    env = EpisodeRecorder(env, run_name)
    return env


def register_custom_env_envs(enjoy=False):
    if enjoy:
        register_env('HomeostaticCrafter-v1', make_custom_env_enjoy)
    else:
        register_env('HomeostaticCrafter-v1', make_custom_env)


def register_modular_hcrafter_models():
    # 2. Register our new modular encoder
    global_model_factory().register_encoder_factory(make_modular_hcrafter_encoder)


def parse_args(argv=None, evaluation=False):
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    hcrafter_override_defaults(parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg


def main():
    register_custom_env_envs()
    # 3. Call the function to register the modular model
    register_modular_hcrafter_models()
    
    cfg = parse_args()
    
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())