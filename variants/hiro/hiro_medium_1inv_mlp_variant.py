import math

from rlkit.policies.network_food import FoodNetworkEasy, FlatFoodNetworkMedium, FoodNetworkMediumFullObs
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.sac.policies import CategoricalPolicy
from torch.nn import functional as F

variant = dict(
    env_name="MiniGrid-Food-8x8-Empty-FullObs-Navigate-v1",
    algorithm="HIRO",
    version="normal",
    layer_size=64,
    replay_buffer_size=int(1E5),
    setter_horizon=250,
    algorithm_kwargs=dict(
        num_epochs=3000,
        num_eval_steps_per_epoch=5000,
        num_low_trains_per_train_loop=1000,
        num_high_trains_per_train_loop=1,
        num_expl_steps_per_train_loop=1000,
        min_num_steps_before_training=1000,
        max_path_length=1000,
        batch_size=256,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        tau=5e-3,
        qf_lr=3E-4,
        setter_lr=3E-4,
        target_setter_noise=0.2,
        target_setter_noise_clip=0.5,
        setter_and_target_update_period=2,
        grad_clip_val=5,

        reward_scale=1,
        use_automatic_entropy_tuning=True,
    )
)
