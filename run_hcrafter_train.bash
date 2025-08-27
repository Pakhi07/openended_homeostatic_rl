python train_hcrafter_env.py \
--env=HomeostaticCrafter-v1 \
--experiment=myhcrafter_Aug_27_1M \
--stats_avg=100 \
--num_envs_per_worker=2 \
--with_wandb=False \
--wandb_user=pakhibanchalia2418 \
--wandb_project=homeostatic_crafter \
--wandb_group=no_maxgrad_norm \
--gamma=0.995 \
--train_for_env_steps=1000000 \
--batch_size=1024 \
# --num_workers=1 \
# --rollout=64 \
# --device=cpu
