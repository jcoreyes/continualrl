from rlkit.policies.network_food import FoodNetworkMedium
from rlkit.torch.conv_networks import CNN
from rlkit.torch.networks import FlattenMlp
from torch.nn import functional as F


variant = dict(
		env_name="MiniGrid-Food-32x32-Medium-10and4-Vision-v1",
		algorithm="SAC Discrete",
		version="normal",
		layer_size=256,
		replay_buffer_size=int(1E4),
		algorithm_kwargs=dict(
			num_epochs=3000,
			num_eval_steps_per_epoch=5000,
			num_trains_per_train_loop=1000,
			num_expl_steps_per_train_loop=1000,
			min_num_steps_before_training=1000,
			max_path_length=3000,
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
		img_conv_kwargs=dict(
			# 7 grid * 8 pixel/grid
			input_width=56,
			input_height=56,
			# 3 rgb channels
			input_channels=3,
			output_size=128,
			kernel_sizes=[5, 3, 3],
			n_channels=[16, 32, 32],
			strides=[3, 2, 2],
			paddings=[0, 0, 0],
			hidden_sizes=[512, 512],
		),
		full_img_conv_kwargs=dict(
			# 32 grid * 2 pixel/grid
			# TODO may need to double resolution
			input_width=64,
			input_height=64,
			# 3 rgb channels
			input_channels=3,
			output_size=128,
			kernel_sizes=[5, 3, 3],
			n_channels=[16, 32, 32],
			strides=[3, 2, 2],
			paddings=[0, 0, 0],
			hidden_sizes=[512, 512],
		),
		inventory_network_kwargs=dict(
			# pantry: 50x5, shelf: 5x5, health:1
			input_size=441,
			output_size=64,
			hidden_sizes=[256, 128],
		)
	)


def gen_network(variant, action_dim, layer_size):
	return FoodNetworkMedium(
		img_network=CNN(**variant['img_conv_kwargs']),
		full_img_network=CNN(**variant['full_img_conv_kwargs']),
		inventory_network=FlattenMlp(**variant['inventory_network_kwargs']),
		final_network=FlattenMlp(
			input_size=variant['img_conv_kwargs']['output_size']
					   + variant['full_img_conv_kwargs']['output_size']
					   + variant['inventory_network_kwargs']['output_size'],
			output_size=action_dim,
			hidden_sizes=[layer_size, layer_size],
			output_activation=F.softmax
		),
		sizes=[
			variant['img_conv_kwargs']['input_width'] * variant['img_conv_kwargs']['input_height'] * variant['img_conv_kwargs']['input_channels'],
			variant['full_img_conv_kwargs']['input_width'] * variant['full_img_conv_kwargs']['input_height'] * variant['full_img_conv_kwargs']['input_channels'],
			# health dim
			1,
			# pantry dim
			400,
			# shelf dim
			40
		]
	)
