"""
Run DQN on grid world.
"""
import copy
import random

from mlagents.envs.exception import UnityWorkerInUseException

from rlkit.envs.unity_envs import MultiDiscreteActionEnv
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
import rlkit.util.hyperparameter as hyp
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
from rlkit.launchers.launcher_util import setup_logger, run_experiment
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm
from gym_unity.envs import UnityEnv
from gym.spaces import Discrete
import numpy as np
from gym import Env
from rlkit.core.logging import get_repo_dir
import os.path as path
from os.path import join


def try_load_unity_env(exec_file, max_attempts=100, no_graphics=True):
    succeeded = False
    env = None
    unity_worker_id = random.randint(400, 1000)
    num_attempts = 0
    while not succeeded and num_attempts <= max_attempts:
        try:
            num_attempts += 1
            env = UnityEnv(exec_file, worker_id=unity_worker_id, use_visual=False, multiagent=False, no_graphics=no_graphics)
            succeeded = True
        except UnityWorkerInUseException as e:
            # try another worker ID
            unity_worker_id = random.randint(400, 1000)
            continue
    if env is None:
        # exited loop due to too many attempts
        raise RuntimeError(
            'Timed out in finding ports to load Unity env at path %s.\nConsider increasing `max_attempts`.' % exec_file
        )
    return env


def experiment(variant):
    # ml_agents_dir = path.join(path.dirname(get_repo_dir()), 'ml-agents') # assume that ml-agents repo is in same dir as continualrl
    env_path = variant['env_path']  # Path to the Unity environment binary to launch
    eval_env = try_load_unity_env(path.join(get_repo_dir(), env_path))
    expl_env = try_load_unity_env(path.join(get_repo_dir(), env_path))
    eval_env = MultiDiscreteActionEnv(eval_env, eval_env.action_space.nvec)
    expl_env = MultiDiscreteActionEnv(expl_env, expl_env.action_space.nvec)
    lifetime = variant.get('lifetime', False)

    obs_dim = expl_env.observation_space.low.size
    action_dim = expl_env.action_space.n

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
    if lifetime:
        eval_policy = expl_policy

    collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
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
    expl_env.close()
    eval_env.close()


if __name__ == "__main__":
    """
        NOTE: Things to check for running exps:
        1. Mode (local vs ec2)
        2. algo_variant, env_variant, env_search_space
        3. use_gpu 
        """
    exp_prefix = 'unity-food-collector-dqn-distincr-resetfree'
    n_seeds = 3
    mode = 'ec2'
    use_gpu = False

    variant = dict(
        algorithm="DQN Lifetime",
        lifetime=True,
        version="distincr - resetfree",
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=200,
            num_eval_steps_per_epoch=5000,
            num_trains_per_train_loop=1000,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=64,
            # validation
            validation_unity_file=join(get_repo_dir(),
                                       'examples/unity/env_builds/dist_increasing/resetfree/FCRFValidationFood2.x86_64'),
            validation_rollout_length=1000,
            validation_period=4
        ),
        trainer_kwargs=dict(
            discount=0.99,
            learning_rate=3E-4,
        )
    )
    search_space = copy.deepcopy(variant)
    search_space = {k: [v] for k, v in search_space.items()}
    search_space.update(
        # insert sweep params here
        env_path=[
            'examples/unity/env_builds/dist_increasing/resetfree/FCRFIncrOver0.x86_64',
            'examples/unity/env_builds/dist_increasing/resetfree/FCRFIncrOver21.x86_64',
            'examples/unity/env_builds/dist_increasing/resetfree/FCRFIncrOver42.x86_64',
            'examples/unity/env_builds/dist_increasing/resetfree/FCRFIncrOver85.x86_64',
            'examples/unity/env_builds/dist_increasing/resetfree/FCRFIncrOver170.x86_64'
        ]
    )

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )

    # setup_logger('unity-food-collector-dqn-test', variant=variant)
    # # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(variant)
    for exp_id, vari in enumerate(sweeper.iterate_hyperparameters()):
        for _ in range(n_seeds):
            run_experiment(
                experiment,
                exp_prefix=exp_prefix,
                mode=mode,
                variant=vari,
                use_gpu=use_gpu,
                region='us-east-2',
                num_exps_per_instance=1,
                snapshot_mode='gap',
                snapshot_gap=10,
                instance_type='c4.4xlarge',
                spot_price=0.20,
                # python_cmd='python3.6'
            )
