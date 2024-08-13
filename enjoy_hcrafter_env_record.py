import sys
import time
from collections import deque
from copy import deepcopy
from os import makedirs

import numpy as np
import torch
from sample_factory.enjoy import visualize_policy_inputs, render_frame
from sample_factory.huggingface.huggingface_utils import generate_replay_video
from sample_factory.utils.utils import experiment_dir

from train_hcrafter_env import parse_args, register_custom_env_envs, register_hcrafter_models

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

max_step = 2_000

register_custom_env_envs(enjoy=True)
register_hcrafter_models()

cfg = parse_args(evaluation=True)

verbose = False

cfg = load_from_checkpoint(cfg)

eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
assert (
        cfg.env_frameskip % eval_env_frameskip == 0
), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip

cfg.num_envs = 1

render_mode = "human"
if cfg.save_video:
    render_mode = "rgb_array"
elif cfg.no_render:
    render_mode = None

env = make_env_func_batched(
    cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
)
env_info = extract_env_info(env, cfg)

if hasattr(env.unwrapped, "reset_on_init"):
    # reset call ruins the demo recording for VizDoom
    env.unwrapped.reset_on_init = False

actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
actor_critic.eval()

device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
actor_critic.model_to_device(device)

policy_id = cfg.policy_index
name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
actor_critic.load_state_dict(checkpoint_dict["model"])


def max_frames_reached(frames):
    return cfg.max_num_frames is not None and frames > cfg.max_num_frames


is_finished = False

num_frames = 0
num_episodes = 0
last_render_start = time.time()
prev_achievement = None
video_frames = []

while is_finished is False:
    # data
    data_json = {
        "step": [],
        "action": [],
        "daylight": [],
        "health": [],
        "hunger": [],
        "thirst": [],
        "energy": [],
        "eat_cow": [],
        "wake_up": [],
        "drink": [],
        "defeat_zombie": [],
    }


    def data_save(a, info):
        global prev_achievement

        data_json["step"].append(info["step"])
        data_json["action"].append(int(a))
        data_json["daylight"].append(float(info["daylight"]))
        data_json["health"].append(float(info["interoception"][0]))
        data_json["hunger"].append(float(info["interoception"][1]))
        data_json["thirst"].append(float(info["interoception"][2]))
        data_json["energy"].append(float(info["interoception"][3]))

        data_json["drink"].append(int(info["achievements"]["collect_drink"] > prev_achievement["collect_drink"]))
        data_json["eat_cow"].append(int(info["achievements"]["eat_cow"] > prev_achievement["eat_cow"]))
        data_json["wake_up"].append(int(info["achievements"]["wake_up"] > prev_achievement["wake_up"]))
        data_json["defeat_zombie"].append(int(info["achievements"]["defeat_zombie"] > prev_achievement["defeat_zombie"]))
        prev_achievement = deepcopy(info["achievements"])


    obs, infos = env.reset()
    prev_achievement = deepcopy(infos[0]["achievements"])

    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False for _ in range(env.num_agents)]

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

            if not cfg.no_render:
                visualize_policy_inputs(normalized_obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs["actions"]

            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)

            # actions shape should be [num_agents, num_actions] even if it's [1, 1]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            # data save
            data_save(a=actions[0], info=infos[0])

            for _ in range(render_action_repeat):
                last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

                obs, rew, terminated, truncated, infos = env.step(actions)
                dones = make_dones(terminated, truncated)

                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.float().clone()
                else:
                    episode_reward += rew.float()

                num_frames += 1
                if num_frames % 100 == 0:
                    print(f"Num frames {num_frames}/{max_step}...")

                dones = dones.cpu().numpy()
                if any(dones):
                    render_frame(cfg, env, video_frames, num_episodes, last_render_start)
                    break

            if num_frames > max_step:
                print("finish!")
                is_finished = True
                break

            if any(dones):
                num_frames = 0
                num_episodes = 0
                last_render_start = time.time()
                prev_achievement = None
                video_frames = []
                print("restart!")
                break

if cfg.save_video:
    if cfg.fps > 0:
        fps = cfg.fps
    else:
        fps = 30
    generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps, cfg)

env.close()

print("save data...")
makedirs("progress_data", exist_ok=True)
np.save(file="progress_data/video_frame.npy", arr=np.asarray(video_frames))

import json
with open("progress_data/progress.json", "w") as outfile:
    json.dump(data_json, outfile)

print("all done. finish.")