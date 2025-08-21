python train_hcrafter_env.py \
--env=HomeostaticCrafter-v1 \
--experiment=myhcrafter \
--stats_avg=100 \
--num_envs_per_worker=10 \
--with_wandb=False \
--wandb_user=pakhibanchalia2418 \
--wandb_project=homeostatic_crafter \
--wandb_group=no_maxgrad_norm \
--gamma=0.995 \
# --device=cpu
