python train_hcrafter_env.py \
--env=HomeostaticCrafter-v1 \
--experiment=modular_debug_run \
--stats_avg=100 \
--num_envs_per_worker=2 \
--with_wandb=False \
--wandb_user=pakhibanchalia2418 \
--wandb_project=homeostatic_crafter \
--wandb_group=no_maxgrad_norm \
--gamma=0.995 \
--train_for_env_steps=10000 \
--num_workers=1 \
--rollout=64 \
--batch_size=256 \
--normalize_input=False \
# --device=cpu
