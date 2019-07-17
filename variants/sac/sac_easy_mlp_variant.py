import math

from rlkit.policies.network_food import FoodNetworkEasy
from rlkit.pythonplusplus import identity
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp, Mlp
from rlkit.torch.sac.policies import CategoricalPolicy
from torch.nn import functional as F


variant = dict(
		# env_name="MiniGrid-Food-8x8-Easy-10and4-Cap50-Decay-v1",
		env_name="MiniGrid-Food-8x8-Easy-Cap50-Decay-v1",
		algorithm="SAC Discrete",
		version="normal",
		layer_size=256,
		replay_buffer_size=int(1E5),
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
			soft_target_tau=5e-3,
			target_update_period=1,
			policy_lr=3E-4,
			qf_lr=3E-4,
			reward_scale=1,
			use_automatic_entropy_tuning=True,
		),
		network_kwargs=dict(
			input_size=227
		)
	)


def gen_network(variant, action_dim, layer_size, policy=False):
	if policy:
		network = CategoricalPolicy(
			Mlp(
				input_size=variant['network_kwargs']['input_size'],
				output_size=action_dim,
				hidden_sizes=[layer_size, layer_size],
				output_activation=F.softmax
			)
		)
	else:
		network = FlattenMlp(
			input_size=variant['network_kwargs']['input_size'],
			output_size=action_dim,
			hidden_sizes=[layer_size, layer_size]
		)

	return network
