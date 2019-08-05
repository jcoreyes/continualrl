"""
Run DQN on grid world.
"""

import gym
from rlkit.policies.ensemble import EnsembleArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn_ensemble import DQNEnsembleTrainer
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import torch.nn as nn


def experiment(variant):
    from rlkit.envs.gym_minigrid.gym_minigrid import envs
    expl_env = gym.make('MiniGrid-UMaze-10x10-v1')
    eval_env = gym.make('MiniGrid-UMaze-10x10-v1')
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    ensemble_size = 5

    qfs = nn.ModuleList([Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
        for _ in range(ensemble_size)])

    target_qfs = nn.ModuleList([Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
        for _ in range(ensemble_size)])

    qf_criterion = nn.MSELoss()
    eval_policy = EnsembleArgmaxDiscretePolicy(qfs)

    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        expl_policy,
    )
    trainer = DQNEnsembleTrainer(
        qfs=qfs,
        target_qfs=target_qfs,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="DQN Ensemble",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        ),
    )
    setup_logger('dqn-ensemble-umaze', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
