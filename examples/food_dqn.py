"""
Run DQN on grid world.
"""

import gym
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.sac.policies import SoftmaxQPolicy
from torch import nn as nn

from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedySchedule
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm
from rlkit.envs.gym_minigrid.gym_minigrid import *
# from variants.dqn_lifetime.dqn_easy_mlp_variant import variant, gen_network
# from variants.dqn_expl.dqn_expl_medium8_mlp_variant import variant, gen_network
from variants.dqn_expl.dqn_expl_medium8_mlp_partial_variant import variant, gen_network


def schedule(t):
    return max(1 - 1e-3 * t, 0.05)


def experiment(variant):
    expl_env = gym.make(variant['env_name'])
    eval_env = gym.make(variant['env_name'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    layer_size = variant['layer_size']
    lifetime = bool(variant.get('lifetime', None))

    qf = gen_network(variant, action_dim, layer_size)
    target_qf = gen_network(variant, action_dim, layer_size)

    qf_criterion = nn.MSELoss()
    # eval_policy = ArgmaxDiscretePolicy(qf)
    eval_policy = SoftmaxQPolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedySchedule(expl_env.action_space, schedule),
        eval_policy,
    )
    collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        # render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        expl_policy,
    )
    trainer = DoubleDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        dtype='int16'
    )
    algo_class = TorchLifetimeRLAlgorithm if lifetime else TorchBatchRLAlgorithm
    algorithm = algo_class(
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
    exp_prefix = 'food-dqn'

    setup_logger(exp_prefix, variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)

    # mode = 'ec2'
    #
    # run_experiment(
    #     experiment,
    #     exp_prefix=exp_prefix,
    #     mode=mode,
    #     variant=variant,
    #     use_gpu=False,
    #     region='us-west-2',
    #     num_exps_per_instance=3
    # )
