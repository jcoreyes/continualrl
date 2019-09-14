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
from variants.minecraft.dqn_dense_navigate import gen_network
import minerl
import argparse
from rlkit.envs.minerl.env_wrappers import wrap_env
from torch.nn import functional as F
# from variants.dqn_lifetime.dqn_medium8_mlp_task_partial_variant import variant, gen_network


# --exclude-keys back sprint sneak jump left right
# 0 nop 1 forward 2 attack 3 left 4 right

def experiment(variant):
    import minerl
    args = variant['args']
    core_env = gym.make(args['env'])
    env = wrap_env(core_env, False, args)

    # eval_env = wrap_env(core_env, True, args)
    #
    # action_dim = expl_env.action_space.n
    # print("Action_dim:", action_dim)


    obs = env.reset()
    steps = 0
    done = False
    print(env._actions)
    total_reward = 0
    while not done and (steps < 1000):
        # action = env.action_space.sample()
        # import pdb; pdb.set_trace()
        #print(env.action_names)
        action = input('Enter action: ')
        key_to_action = dict(p=0, w=1, s=2, a=3, d=4)
        #key_to_action = dict(s=0, a=1, d=2)
        if action not in key_to_action:
            continue
        obs, reward, done, info = env.step(key_to_action[action])
        total_reward += reward
        print(reward, total_reward)
        env.render()
        steps += 1

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
                        # Added
                        'MineRLEating-v0',
                        'MineRLMazeRunner-v0',
                        'MineRLShapingTreechop-v0'
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
variant = dict()
variant['args'] = args
experiment(variant)