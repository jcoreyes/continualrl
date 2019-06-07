from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, Food


class FoodEnv(FoodEnvBase):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def extra_step(self, action):
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

	def place_food(self):
		if self.step_count % self.food_rate == 0:
			self.place_obj(Food())

	def decay_health(self):
		if self.step_count and self.step_count % self.health_rate == 0:
			self.add_health(-1)


class FoodEnv6and4(FoodEnv):
	def __init__(self):
		super().__init__(health_rate=6)


class FoodEnv10and4(FoodEnv):
	def __init__(self):
		super().__init__(health_rate=10)


register(
	id='MiniGrid-Food-8x8-4and4-v1',
	entry_point='gym_minigrid.envs:FoodEnv'
)

register(
	id='MiniGrid-Food-8x8-6and4-v1',
	entry_point='gym_minigrid.envs:FoodEnv6and4'
)

register(
	id='MiniGrid-Food-8x8-10and4-v1',
	entry_point='gym_minigrid.envs:FoodEnv10and4'
)
