from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, Food


class FoodEnvEasy(FoodEnvBase):
	"""
	Pick up food to gain 1 health point,
	Lose 1 health point every `health_rate` timesteps,
	Get 1 reward per timestep
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=(12481,) if self.obs_vision else (227,),
			dtype='uint8'
		)

	def extra_step(self, action, matched):
		if matched:
			return matched

		agent_cell = self.grid.get(*self.agent_pos)
		matched = True

		# Collect resources. In the case of this env, mining = instant health bonus.
		if action == self.actions.mine:
			if agent_cell and agent_cell.can_mine(self):
				self.grid.set(*self.agent_pos, None)
				self.add_health(agent_cell.food_value())
		else:
			matched = False

		return matched

	def place_items(self):
		if self.step_count % self.food_rate == 0:
			self.place_obj(Food())


class FoodEnvEasy6and4(FoodEnvEasy):
	def __init__(self):
		super().__init__(health_rate=6)


class FoodEnvEasy6and4Vision(FoodEnvEasy):
	def __init__(self):
		super().__init__(health_rate=6, obs_vision=True)


class FoodEnvEasy10and4(FoodEnvEasy):
	def __init__(self):
		super().__init__(health_rate=10)


class FoodEnvEasy10and4Vision(FoodEnvEasy):
	def __init__(self):
		super().__init__(health_rate=10, obs_vision=True)


register(
	id='MiniGrid-Food-8x8-Easy-4and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvEasy'
)

register(
	id='MiniGrid-Food-8x8-Easy-6and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvEasy6and4'
)

register(
	id='MiniGrid-Food-8x8-Easy-6and4-Vision-v1',
	entry_point='gym_minigrid.envs:FoodEnvEasy6and4Vision'
)

register(
	id='MiniGrid-Food-8x8-Easy-10and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvEasy10and4'
)

register(
	id='MiniGrid-Food-8x8-Easy-10and4-Vision-v1',
	entry_point='gym_minigrid.envs:FoodEnvEasy10and4Vision'
)