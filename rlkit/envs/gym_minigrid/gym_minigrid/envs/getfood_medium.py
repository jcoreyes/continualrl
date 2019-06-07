from enum import IntEnum

from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, MiniGridAbsoluteEnv, DIR_TO_VEC, \
	GridAbsolute


class FoodEnvMedium(FoodEnvBase):
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
		# place objects down
		place0 = 6
		place1 = 7
		place2 = 8
		place3 = 9
		place4 = 10

	def __init__(
			self,
			health_cap=10,
			health_rate=4,
			food_rate=4
	):
		super().__init__(
			grid_size=32,
			health_cap=health_cap,
			health_rate=health_rate,
			food_rate=food_rate
		)

		self.actions = FoodEnvMedium.Actions
		# food
		self.pantry = []
		# other resources
		self.shelf = [None] * 5
		self.shelf_type = [''] * 5
		self.interactions = {
			('energy', 'metal') : Axe,
			('axe', 'tree')     : Wood
		}

	def place_food(self):
		if self.step_count % self.food_rate == 0:
			self.place_obj(Food())
		if self.step_count % (2 * self.food_rate) == 0:
			self.place_obj(Metal())
			self.place_obj(Energy())
		if self.step_count % (3 * self.food_rate) == 0:
			self.place_obj(Tree())

	def decay_health(self):
		if self.step_count and self.step_count % self.health_rate == 0:
			self.add_health(-1)

	def extra_step(self, action):
		agent_cell = self.grid.get(*self.agent_pos)
		matched = True

		# Collect resources. Add to shelf.
		if action == self.actions.mine:
			if agent_cell and agent_cell.can_mine(self):
				mined = False
				# check if food or other resource, which we're storing separately
				if agent_cell.food_value() > 0:
					self.pantry.append(agent_cell)
					mined = True
				else:
					mined = self.add_to_shelf(agent_cell)

				if mined:
					self.grid.set(*self.agent_pos, None)

		# Consume stored food.
		elif action == self.actions.eat:
			self.pantry.sort(key=lambda item: item.food_value(), reverse=True)
			if self.pantry:
				eaten = self.pantry.pop(0)
				self.add_health(eaten.food_value())

		# actions to use each element of inventory
		elif action == self.actions.place0:
			self.place_act(0)

		elif action == self.actions.place1:
			self.place_act(1)

		elif action == self.actions.place2:
			self.place_act(2)

		elif action == self.actions.place3:
			self.place_act(3)

		elif action == self.actions.place4:
			self.place_act(4)

		else:
			matched = False

		return matched

	def place_act(self, action):
		obj = self.shelf[action]
		obj_type = self.shelf_type[action]
		agent_cell = self.grid.get(*self.agent_pos)
		if obj is None:
			# there's nothing to place
			return
		elif agent_cell is None:
			# there's nothing to combine it with, so just place it on the grid
			self.grid.set(*self.agent_pos, obj)
		else:
			# let's try to combine the placed object with the existing object
			interact_tup = tuple(sorted([obj_type, agent_cell.type]))
			new_obj = self.interactions.get(interact_tup, None)
			if not new_obj:
				# the objects cannot be combined, no-op
				return
			else:
				# replace existing obj with new obj
				self.grid.set(*self.agent_pos, new_obj())
		# remove placed object from inventory
		self.shelf[action] = None
		self.shelf_type[action] = ''

	def add_to_shelf(self, obj):
		""" Returns whether adding to shelf succeeded """
		if None in self.shelf:
			idx = self.shelf.index(None)
			self.shelf[idx] = obj
			self.shelf_type[idx] = obj.type
			return True
		return False

	def can_cut(self):
		""" Can we cut objects? """
		return 'axe' in self.shelf_type


class FoodEnvMedium6and4(FoodEnvMedium):
	def __init__(self):
		super().__init__(health_rate=6)


class FoodEnvMedium10and4(FoodEnvMedium):
	def __init__(self):
		super().__init__(health_rate=10)


register(
	id='MiniGrid-Food-8x8-Medium-4and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvMedium'
)

register(
	id='MiniGrid-Food-8x8-Medium-6and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvMedium6and4'
)

register(
	id='MiniGrid-Food-8x8-Medium-10and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvMedium10and4'
)
