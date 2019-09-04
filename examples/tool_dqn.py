"""
Run DQN on grid world.
"""

import gym
from gym_minigrid.envs.tools import ToolsEnv
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector, MdpPathCollectorConfig
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.sac.policies import SoftmaxQPolicy
from torch import nn as nn
import rlkit.util.hyperparameter as hyp
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedySchedule, EpsilonGreedyDecay
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.networks import Mlp
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm

from variants.dqn.dqn_medium_mlp_task_partial_variant import variant as algo_variant, gen_network
# from variants.dqn_lifetime.dqn_medium8_mlp_task_partial_variant import variant as algo_variant, gen_network


def schedule(t):
    print(t)
    return max(1 - 5e-4 * t, 0.05)


def experiment(variant):
    from rlkit.envs.gym_minigrid.gym_minigrid import envs

    expl_env = ToolsEnv(
        **variant['env_kwargs']
    )
    eval_env = ToolsEnv(
        **variant['env_kwargs']
    )
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    layer_size = variant['layer_size']
    lifetime = variant.get('lifetime', False)
    if lifetime:
        assert eval_env.time_horizon == 0, 'cannot have time horizon for lifetime env'

    qf = gen_network(variant, action_dim, layer_size)
    target_qf = gen_network(variant, action_dim, layer_size)

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    # eval_policy = SoftmaxQPolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedyDecay(expl_env.action_space, 1e-4, 1, 0.1),
        eval_policy,
    )
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     EpsilonGreedy(expl_env.action_space, 0.5),
    #     eval_policy,
    # )
    if eval_env.time_horizon == 0:
        collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    else:
        collector_class = MdpPathCollectorConfig
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        # render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        expl_policy,
        # render=lifetime
        # render=True
    )
    trainer = DoubleDQNTrainer(
        qf=qf,
        target_qf=target_qf,
        qf_criterion=qf_criterion,
        **variant['trainer_kwargs']
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env
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
    from variants.exps.dynamic_static_reset_free.env_search_space import env_search_space
    from variants.envs.axe_reset import env_variant

    exp_prefix = 'tool-dqn-prob-time-sweep'

    n_seeds = 1
    mode = 'ec2'

    # Comment below to run sweep, uncomment to run default variant
    env_search_space = {}

    env_sweeper = hyp.DeterministicHyperparameterSweeper(
        env_search_space, default_parameters=env_variant,
    )

    for exp_id, env_vari in enumerate(env_sweeper.iterate_hyperparameters()):
        variant = dict(algo_variant)
        assert 'env_kwargs' not in variant, '`env_kwargs` in variant will get overridden, should be in separate variant'
        variant['env_kwargs'] = env_vari
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=variant,
                use_gpu=False,
                region='us-west-2',
                num_exps_per_instance=3
            )
