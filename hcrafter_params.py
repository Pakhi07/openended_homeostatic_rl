import os
from os.path import join

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import str2bool



def add_hcrafter_env_eval_args(parser):
    """Arguments used only during evaluation."""
    parser.add_argument(
        "--record_to",
        # default=join(os.getcwd(), "..", "recs"),
        default=None,
        type=str,
        help="Record episodes to this folder. This records a demo that can be replayed at full resolution. Currently, this does not work for bot environments so it is recommended to use --save_video to record episodes at lower resolution instead for such environments",
    )


def hcrafter_override_defaults(parser):
    """RL params specific to Doom envs."""
    parser.set_defaults(
        ppo_clip_value=0.2,  # value used in all experiments in the paper
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        exploration_loss="symmetric_kl",
        exploration_loss_coeff=0.001,
        normalize_returns=True,
        normalize_input=True,
        # max_grad_norm=10.0,
        env_frameskip=1,
        eval_env_frameskip=1,  # this is for smoother rendering during evaluation
        fps=0,  # for evaluation only
        num_workers=50,
        heartbeat_reporting_interval=600,
        use_rnn=True,
        rnn_size=512,
        rnn_num_layers=1,
        rnn_type="gru",
        encoder_conv_architecture="resnet_impala",
        batch_size=1024,
        train_for_env_steps=5_000_000_000,
    )


def default_hcrafter_cfg(algo="APPO", env="env", experiment="test"):
    """Useful in tests."""
    argv = [f"--algo={algo}", f"--env={env}", f"--experiment={experiment}"]
    parser, args = parse_sf_args(argv)
    hcrafter_override_defaults(parser)
    args = parse_full_cfg(parser, argv)
    return args
