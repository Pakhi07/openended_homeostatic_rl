import sys

from sample_factory.enjoy import enjoy
from train_hcrafter_env import parse_args, register_custom_env_envs, register_hcrafter_models


def main():
    register_custom_env_envs(enjoy=True)
    register_hcrafter_models()

    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
