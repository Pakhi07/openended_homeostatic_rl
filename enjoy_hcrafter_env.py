import sys

from sample_factory.enjoy import enjoy
from train_hcrafter_env import parse_args, register_custom_env_envs, register_hcrafter_models

import torch

# Force torch.load to always use weights_only=False
_old_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _old_load(*args, **kwargs)

torch.load = _patched_load

def main():
    register_custom_env_envs(enjoy=True)
    register_hcrafter_models()

    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
