"""
Run SAC on Swimmer Gather Mujoco task.
"""
import math
from os.path import join

import gym
import copy

#from gym_minigrid.envs.deer_diverse import DeerDiverseEnv
#from gym_minigrid.envs.lava import LavaEnv
#from gym_minigrid.envs.monsters import MonstersEnv
#from gym_minigrid.envs.tools import ToolsEnv
from rlkit.core.logging import get_repo_dir
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector, MdpPathCollectorConfig
from rlkit.torch.sac.policies import SoftmaxQPolicy, TanhGaussianPolicy, MakeDeterministic
from torch import nn as nn

from rlkit.torch.sac.sac import SACTrainer
import rlkit.util.hyperparameter as hyp
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedySchedule, EpsilonGreedyDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp, FlattenMlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm
from rlkit.envs.swimmer_gather_mujoco.swimmer_gather_env import SwimmerGatherEnv

def experiment(variant):
    #if variant['env_kwargs']['action_noise_mode'] is None:
    #    if variant['env_kwargs']['action_noise_std'] != 0.03 or variant['env_kwargs']['action_noise_discount'] != 0.9:
    #        return
    expl_env = SwimmerGatherEnv(**variant['env_kwargs'])
    eval_env = SwimmerGatherEnv(**variant['env_kwargs'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    lifetime = variant.get('lifetime', False)

    M = variant['algo_kwargs']['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = MakeDeterministic(policy)

    collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        # render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        policy,
        # render=True
    )
    replay_buffer = EnvReplayBuffer(
        variant['algo_kwargs']['replay_buffer_size'],
        expl_env,
        dtype='float16'
    )
    trainer = SACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['algo_kwargs']['trainer_kwargs']
    )

    algo_class = TorchLifetimeRLAlgorithm if lifetime else TorchBatchRLAlgorithm
    algorithm = algo_class(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algo_kwargs']['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    """
    NOTE: Things to check for running exps:
    1. Mode (local vs ec2)
    2. algo_variant, env_variant, env_search_space
    3. use_gpu 
    """
    exp_prefix = 'swimmer-gather-envshaping'
    n_seeds = 1
    mode = 'ec2'
    use_gpu = False

    env_variant = dict(
        ball_radius=0.5,
        radius=100,
        radius_decay=0,
        radius_mode='ball',

        action_noise_std=0.1,
        action_noise_discount=0.98,
        action_noise_mode='average',
    )
    env_search_space = copy.deepcopy(env_variant)
    env_search_space = {k: [v] for k, v in env_search_space.items()}
    #env_search_space.update(
    #    # TODO find scale of env for radius
    #    radius=[0.5, 1, 2],
    #    # TODO adjust radius decay accordingly
    #    radius_decay=[1e-6, 1e-5, 1e-4],
    #    radius_mode=['ball', 'circle'],

    #    # TODO find scale of actions
    #    action_noise_std=[0.03, 0.1, 0.3],
    #    action_noise_discount=[0.9, 0.95, 0.98],  # effective dynamism time-horizon of (1-discount)^{-1} timesteps
    #    action_noise_mode=[None, 'average', 'walk'],
    #    time_horizon=[0, 200]
    #)

    algo_variant = dict(
        algorithm="SAC",
        lifetime=True,
        version="swimmer gather - env shaping",
        layer_size=16,
        replay_buffer_size=int(5E5),
        eps_decay_rate=1e-5,
        algorithm_kwargs=dict(
            num_epochs=1500,
            num_eval_steps_per_epoch=6000,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=200,
            max_path_length=math.inf,
            batch_size=64,
            # validation_envs_pkl=join(get_repo_dir(), 'examples/continual/env_shaping/diverse_deer/validation_envs/env_shaping_validation_envs_2020_02_05_06_58_29.pkl'),
            # validation_rollout_length=200,
            # validation_period=10,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
        inventory_network_kwargs=dict(
            # shelf: 8 x 8
            input_size=64,
            output_size=16,
            hidden_sizes=[16, 16],
        ),
        full_img_network_kwargs=dict(
            # 5 x 5 x 8
            input_size=200,
            output_size=32,
            hidden_sizes=[64, 64]
        ),
        num_obj_network_kwargs=dict(
            # num_objs: 8
            input_size=8,
            output_size=8,
            hidden_sizes=[8]
        )
    )
    algo_search_space = copy.deepcopy(algo_variant)
    algo_search_space = {k: [v] for k, v in algo_search_space.items()}
    algo_search_space.update(
        # insert sweep params here
    )

    env_sweeper = hyp.DeterministicHyperparameterSweeper(
        env_search_space, default_parameters=env_variant,
    )
    algo_sweeper = hyp.DeterministicHyperparameterSweeper(
        algo_search_space, default_parameters=algo_variant,
    )

    for exp_id, env_vari in enumerate(env_sweeper.iterate_hyperparameters()):
        for algo_vari in algo_sweeper.iterate_hyperparameters():
            variant = {'algo_kwargs': algo_vari, 'env_kwargs': env_vari}
            for _ in range(n_seeds):
                run_experiment(
                    experiment,
                    exp_prefix=exp_prefix,
                    mode=mode,
                    variant=variant,
                    use_gpu=use_gpu,
                    region='us-east-2',
                    #num_exps_per_instance=3, # 3
                    num_exps_per_instance=1, # 3
                    snapshot_mode='gap',
                    snapshot_gap=10,
                    instance_type='c5.xlarge',
                    spot_price=0.07,
                #    python_cmd='python3'
                )
                break
