"""
Run DQN on grid world.
"""

import gym
import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.exploration_strategies.base import \
    PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedyDecay
from rlkit.launchers.launcher_util import run_experiment
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm
from torch import nn as nn
# from variants.dqn.dqn_medium8_mlp_task_variant import variant, gen_network
from variants.minecraft.dqn_dense_navigate import variant, gen_network
import minerl
import argparse
from rlkit.envs.minerl.env_wrappers import wrap_env
# from variants.dqn_lifetime.dqn_medium8_mlp_task_partial_variant import variant, gen_network

def schedule(t):
    return max(1 - 5e-4 * t, 0.05)


def experiment(variant):
    import minerl
    args = variant['args']
    core_env = gym.make(args['env'])
    expl_env = wrap_env(core_env, False, args)

    eval_env = wrap_env(core_env, True, args)

    action_dim = expl_env.action_space.n

    lifetime = variant.get('lifetime', False)

    qf = gen_network(variant, action_dim)
    target_qf = gen_network(variant, action_dim)

    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
    # eval_policy = SoftmaxQPolicy(qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedyDecay(expl_env.action_space, 2e-4, 1, 0.1),
        eval_policy,
    )
    # expl_policy = PolicyWrappedWithExplorationStrategy(
    #     EpsilonGreedy(expl_env.action_space, 0.5),
    #     eval_policy,
    # )
    collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        expl_policy,
        render=lifetime
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


# python examples/minecraft/dqn_test.py --env MineRLNavigateDense-v0 --always-keys forward sprint
# attack --exclude-keys back left right sneak place --frame-stack 4 --frame-skip 4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MineRLTreechop-v0',
                        choices=[
                            'MineRLTreechop-v0',
                            'MineRLNavigate-v0', 'MineRLNavigateDense-v0',
                            'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
                            'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
                            'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
                            # for debug use
                            'MineRLNavigateDenseFixed-v0',
                            'MineRLObtainTest-v0',
                        ],
                        help='MineRL environment identifier.')
    parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--disable-action-prior', action='store_true', default=False,
                        help='If specified, action_space shaping based on prior knowledge will be disabled.')
    parser.add_argument('--always-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed throughout interaction with environment.')
    parser.add_argument('--reverse-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be always pressed but can be turn off via action.')
    parser.add_argument('--exclude-keys', type=str, default=None, nargs='*',
                        help='List of action keys, which should be ignored for discretizing action space.')
    parser.add_argument('--exclude-noop', action='store_true', default=False, help='The "noop" will be excluded from discrete action list.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001,
                        help='Exploration epsilon used during eval episodes.')
    args = vars(parser.parse_args())

    exp_prefix = 'minecraft-navigate-dense'

    # setup_logger(exp_prefix, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(variant)

    mode = 'local'
    variant['args'] = args
    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
        region='us-west-2',
        num_exps_per_instance=3
    )
