from gym.envs.mujoco import HalfCheetahEnv

import gym
from rlkit.policies.network_food import FoodNetworkMedium
from rlkit.torch.conv_networks import CNN
from rlkit.torch.sac.sac_discrete import SACDiscreteTrainer
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, CategoricalPolicy, \
    RandomPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.envs.gym_minigrid.gym_minigrid import *

# TODO NOTE: this is where you pick the variant
from variants.sac.sac_easy_mlp_variant import variant, gen_network


def experiment(variant):
    assert gen_network is not None, "unable to import gen_network function, check env variant."

    expl_env = gym.make(variant['env_name'])
    eval_env = gym.make(variant['env_name'])
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.n

    layer_size = variant['layer_size']
    qf1 = gen_network(variant, action_dim, layer_size)
    qf2 = gen_network(variant, action_dim, layer_size)
    target_qf1 = gen_network(variant, action_dim, layer_size)
    target_qf2 = gen_network(variant, action_dim, layer_size)
    policy = gen_network(variant, action_dim, layer_size, policy=True)
    expl_policy = RandomPolicy(action_dim)

    # Use GPU
    if ptu.gpu_enabled():
        qf1 = qf1.cuda()
        qf2 = qf2.cuda()
        target_qf1 = target_qf1.cuda()
        target_qf2 = target_qf2.cuda()
        policy = policy.cuda()

    # eval_policy = MakeDeterministic(policy)
    eval_policy = policy
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        # TODO: can change this to `policy` to switch back to non-random exploration policy
        expl_policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
        dtype='uint8'
    )

    trainer = SACDiscreteTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
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
    setup_logger('sac-discrete', variant=variant)
    ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
    experiment(variant)
