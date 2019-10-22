"""
Run DQN on grid world.
"""

import gym
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
from gym_unity.envs import UnityEnv
from gym.spaces import Discrete
import numpy as np
from gym import Env
from rlkit.core.logging import get_repo_dir
import os.path as path

class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)

class MultiDiscreteActionEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, nvec):
        super().__init__(wrapped_env)
        self.nvec = nvec
        self.multi_actions = np.zeros((np.prod(nvec), nvec.shape[0]))
        self.action_space = Discrete(self.multi_actions.shape[0])

        # iterate over all possible combinations of actions by doing addition in different space
        counter = np.zeros(nvec.shape[0])
        for i in range(self.multi_actions.shape[0]):
            while True:
                if (counter >= nvec).sum() == 0:
                    break
                else:
                    idx = (counter == nvec).argmax()
                    counter[idx] = 0
                    counter[idx-1] += 1
            self.multi_actions[i] = counter
            counter[-1] += 1

    def step(self, action):
        multi_action = self.multi_actions[action]
        state, reward, done, info = super().step(multi_action)
        #import pdb; pdb.set_trace()
        return state, reward, done, {}


def experiment(variant):

    ml_agents_dir = path.join(path.dirname(get_repo_dir()), 'ml-agents') # assume that ml-agents repo is in same dir as continualrl
    env_build_dir = path.join(ml_agents_dir, 'env_builds')
    env_name = "food_collector"  # Name of the Unity environment binary to launch
    env = UnityEnv(path.join(env_build_dir, env_name), worker_id=11, use_visual=False, multiagent=False)
    env = MultiDiscreteActionEnv(env, env.action_space.nvec)
    expl_env = env
    eval_env = env

    #expl_env = gym.make('CartPole-v0')
    #eval_env = gym.make('CartPole-v0')

    obs_dim = expl_env.observation_space.low.size
    action_dim = env.action_space.n

    qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    target_qf = Mlp(
        hidden_sizes=[32, 32],
        input_size=obs_dim,
        output_size=action_dim,
    )
    qf_criterion = nn.MSELoss()
    eval_policy = ArgmaxDiscretePolicy(qf)
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
    trainer = DQNTrainer(
        qf=qf,
        target_qf=target_qf,
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
        algorithm="DQN",
        version="normal",
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
    setup_logger('unity-food-collector-dqn-test', variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
