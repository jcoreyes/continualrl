from gym.envs.mujoco import HalfCheetahEnv

import gym
from rlkit.policies.network_food import FoodNetworkMedium
from rlkit.samplers.data_collector.path_collector import LifetimeMdpPathCollector
from rlkit.torch.conv_networks import CNN
from rlkit.torch.dqn.double_dqn import DoubleDQNTrainer
from rlkit.torch.dqn.dqn import DQNTrainer
from rlkit.torch.sac.sac_discrete import SACDiscreteTrainer
from torch.nn import functional as F

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic, CategoricalPolicy, RandomPolicy, \
	SoftmaxQPolicy
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm, TorchLifetimeRLAlgorithm
from rlkit.envs.gym_minigrid.gym_minigrid import *
import torch.nn as nn

# TODO NOTE: this is where you pick the variant
# from variants.dqn_expl.dqn_expl_easy_grid_conv_variant import variant, gen_network
from variants.dqn_lifetime.dqn_easy_mlp_variant import variant, gen_network


def experiment(variant):
	assert gen_network is not None, "unable to import gen_network function, check env variant."

	expl_env = gym.make(variant['env_name'])
	eval_env = gym.make(variant['env_name'])
	obs_dim = expl_env.observation_space.low.size
	action_dim = eval_env.action_space.n
	layer_size = variant['layer_size']
	lifetime = bool(variant.get('lifetime', None))

	qf = gen_network(variant, action_dim, layer_size)
	target_qf = gen_network(variant, action_dim, layer_size)
	policy = SoftmaxQPolicy(qf)
	expl_policy = RandomPolicy(action_dim)

	# Use GPU
	if ptu.gpu_enabled():
		qf = qf.cuda()
		target_qf = target_qf.cuda()
		policy = policy.cuda()

	# eval_policy = MakeDeterministic(policy)
	eval_policy = policy
	collector_class = LifetimeMdpPathCollector if lifetime else MdpPathCollector
	eval_path_collector = collector_class(
		eval_env,
		eval_policy,
	)
	expl_path_collector = collector_class(
		expl_env,
		# TODO: can change this to `policy` to switch back to non-random exploration policy
		expl_policy,
	)
	replay_buffer = EnvReplayBuffer(
		variant['replay_buffer_size'],
		expl_env,
		dtype='uint8'
	)

	trainer = DoubleDQNTrainer(
		qf=qf,
		target_qf=target_qf,
		qf_criterion=nn.MSELoss(),
		**variant['trainer_kwargs']
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
	# noinspection PyTypeChecker
	setup_logger('dqn-expl', variant=variant)
	ptu.set_gpu_mode(True)  # optionally set the GPU (default=False)
	experiment(variant)
