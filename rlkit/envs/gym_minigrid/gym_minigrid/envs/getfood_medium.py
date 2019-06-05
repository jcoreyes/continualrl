from enum import IntEnum

from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, MiniGridAbsoluteEnv, DIR_TO_VEC, \
	GridAbsolute


class FoodEnv(FoodEnvBase):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""

	class Actions(IntEnum):
		# Absolute directions
		west = 0
		east = 1
		north = 2
		south = 3
		# collect (but don't consume) an item
		mine = 4
		# consume a stored food item to boost health (does nothing if no stored food)
		eat = 5


	def __init__(
			self,
			health_cap=5,
			health_rate=4,
			food_rate=4
	):
		super().__init__(
			health_cap=health_cap,
			health_rate=health_rate,
			food_rate=food_rate
		)

		self.actions = FoodEnv.Actions
		# food
		self.pantry = []
		self.pantry_str = []
		# other resources
		self.shelf = []
		self.shelf_str = []

	def extra_step(self, action):
		agent_cell = self.grid.get(*self.agent_pos)
		matched = True

		# Collect resources. Add to shelf.
		if action == self.actions.mine:
			if agent_cell and agent_cell.can_mine(self):
				self.grid.set(*self.agent_pos, None)
				# check if food or other resource, which we're storing separately
				if agent_cell.food_value() > 0:
					self.pantry.append(agent_cell)
					self.pantry_str.append(str(agent_cell))
				else:
					self.shelf.append(agent_cell)
					self.shelf_str.append(str(agent_cell))

		# Consume stored food.
		elif action == self.actions.eat:
			self.pantry.sort(key=lambda item: item.food_value(), reverse=True)
			if self.pantry:
				eaten = self.pantry.pop(0)
				self.add_health(eaten.food_value())
				self.pantry_str.remove(str(eaten))

		else:
			matched = False

		return matched


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
