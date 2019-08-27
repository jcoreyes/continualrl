import gym
from rlkit.core.hierarchical_batch_rl_algorithm import HierarchicalBatchRLAlgorithm
from rlkit.data_management.goal_replay_buffer import GoalConditionedReplayBuffer, GoalSetterReplayBuffer
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.epsilon_greedy import EpsilonGreedy, EpsilonGreedyDecay, HIROEpsilonGreedyDecay
from rlkit.launchers.goal_setter import GoalSetterNetwork, GoalSetterWrappedWithExplorationStrategy
from rlkit.policies.argmax import ArgmaxDiscretePolicy
from rlkit.samplers.data_collector.path_collector import HierarchicalPathCollector, \
    LifetimeHierarchicalPathCollector
from rlkit.torch.hierarchy import HierarchyTrainer

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.launchers.launcher_util import run_experiment
from rlkit.torch.networks import FlattenMlp, TanhMlpPolicy
from rlkit.torch.torch_rl_algorithm import TorchLifetimeRLAlgorithm

# TODO NOTE: this is where you pick the variant
from variants.hiro.hiro_medium_1inv_mlp_variant import variant


def experiment(variant):
    from rlkit.envs.gym_minigrid.gym_minigrid import envs

    expl_env = gym.make(variant['env_name'])
    eval_env = gym.make(variant['env_name'])
    obs_dim = eval_env.observation_space.low.size
    action_dim = eval_env.action_space.n
    goal_dim = eval_env.goal_space.low.size
    lifetime = bool(variant.get('lifetime', None))

    M = variant['layer_size']
    # goal_dim is the action dim for these networks
    qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=1,
        hidden_sizes=[M, M],
    )
    eval_setter = GoalSetterNetwork(
        policy=TanhMlpPolicy(
            input_size=obs_dim,
            output_size=goal_dim,
            hidden_sizes=[M, M]
        ),
        horizon=variant['setter_horizon']
    )
    expl_setter = GoalSetterWrappedWithExplorationStrategy(
        exploration_strategy=HIROEpsilonGreedyDecay(eval_env.goal_space, 1e-4, 1, 0.1),
        setter=eval_setter,
    )
    target_setter = GoalSetterNetwork(
        policy=TanhMlpPolicy(
            input_size=obs_dim,
            output_size=goal_dim,
            hidden_sizes=[M, M]
        ),
        horizon=variant['setter_horizon']
    )

    low_qf = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    low_target_qf = FlattenMlp(
        input_size=obs_dim + goal_dim,
        output_size=action_dim,
        hidden_sizes=[M, M],
    )
    eval_policy = ArgmaxDiscretePolicy(low_qf)
    expl_policy = PolicyWrappedWithExplorationStrategy(
        EpsilonGreedy(expl_env.action_space),
        eval_policy,
    )

    # Use GPU
    if ptu.gpu_enabled():
        qf1 = qf1.cuda()
        qf2 = qf2.cuda()
        target_qf1 = target_qf1.cuda()
        target_qf2 = target_qf2.cuda()
        eval_setter = eval_setter.cuda()
        target_setter = target_setter.cuda()
        low_qf = low_qf.cuda()
        low_target_qf = low_target_qf.cuda()

    collector_class = LifetimeHierarchicalPathCollector if lifetime else HierarchicalPathCollector
    eval_path_collector = collector_class(
        eval_env,
        eval_policy,
        eval_setter,
        render=True
    )
    expl_path_collector = collector_class(
        expl_env,
        expl_policy,
        expl_setter
    )
    low_replay_buffer = GoalConditionedReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        dtype='float16'
    )
    high_replay_buffer = GoalSetterReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        goal_period=variant['setter_horizon'],
        dtype='float16'
    )
    trainer = HierarchyTrainer(
        env=eval_env,
        setter=eval_setter,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        target_setter=target_setter,
        policy=eval_policy,
        low_qf=low_qf,
        low_target_qf=low_target_qf,
        **variant['trainer_kwargs']
    )

    algo_class = TorchLifetimeRLAlgorithm if lifetime else HierarchicalBatchRLAlgorithm
    algorithm = algo_class(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        low_replay_buffer=low_replay_buffer,
        high_replay_buffer=high_replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    exp_prefix = 'hiro'
    # noinspection PyTypeChecker
    # setup_logger(exp_prefix, variant=variant)
    # ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    # experiment(variant)

    mode = 'local'

    run_experiment(
        experiment,
        exp_prefix=exp_prefix,
        mode=mode,
        variant=variant,
        use_gpu=True,
        region='us-west-2',
        num_exps_per_instance=3,
        snapshot_mode='all'
    )
