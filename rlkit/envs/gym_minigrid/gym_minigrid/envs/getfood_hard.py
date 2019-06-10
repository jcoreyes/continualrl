from enum import IntEnum

from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, MiniGridAbsoluteEnv, DIR_TO_VEC, \
	GridAbsolute


class FoodEnvHard(FoodEnvBase):
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
			food_rate=4,
			max_pantry_size=50,
			obs_vision=False,
			**kwargs
	):
		# food
		self.pantry = []
		self.max_pantry_size = max_pantry_size
		# other resources
		self.shelf = [None] * 5
		self.shelf_type = [''] * 5
		self.interactions = {
			('energy', 'metal'): Axe,
			('energy', 'energy'): BigFood,
			# inedible wood, to be used for shelter
			('axe', 'tree'): Wood
		}
		self.monsters = []
		self.object_to_idx = {
			'empty': 0,
			'wall': 1,
			'food': 2,
			'tree': 3,
			'metal': 4,
			'energy': 5,
			'axe': 6,
			'wood': 7,
			'bigfood': 8,
			'monster': 9
		}
		self.actions = FoodEnvHard.Actions
		super().__init__(
			grid_size=32,
			health_cap=health_cap,
			health_rate=health_rate,
			food_rate=food_rate,
			obs_vision=obs_vision
		)

		if self.obs_vision:
			self.observation_space = spaces.Box(
				low=0,
				high=255,
				shape=(22137,),
				dtype='uint8'
			)

	def place_items(self):
		if self.step_count % self.food_rate == 0:
			self.place_obj(Food())
		if self.step_count % (2 * self.food_rate) == 0:
			self.place_obj(Metal())
			self.place_obj(Energy())
		if self.step_count % (3 * self.food_rate) == 0:
			self.place_obj(Tree())
		if self.step_count % (8 * self.food_rate) == 0:
			# could have parameter for a monster that gets more efficient over time
			# pick random position that isn't in gray border
			monster = Monster(np.array(self._rand_pos(1, self.grid_size-2, 1, self.grid_size-2)),
			                  self,
							  12 * self.food_rate,
							  0.3)
			self.monsters.append(monster)

	def extra_step(self, action, matched):
		# Let monsters act and die
		dead_monsters = []
		for monster in self.monsters:
			if not monster.act(self.last_agent_pos):
				dead_monsters.append(monster)
		for monster in dead_monsters:
			self.monsters.remove(monster)

		if matched:
			return matched

		agent_cell = self.grid.get(*self.agent_pos)
		matched = True
		# Collect resources. Add to shelf.
		if action == self.actions.mine:
			if agent_cell and agent_cell.can_mine(self):
				mined = False
				# check if food or other resource, which we're storing separately
				if agent_cell.food_value() > 0:
					if len(self.pantry) < self.max_pantry_size:
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

	def gen_pantry_obs(self):
		pantry_obs = np.zeros((self.max_pantry_size, len(self.object_to_idx)), dtype=np.uint8)
		pantry_idxs = [self.object_to_idx[obj.type] for obj in self.pantry]
		pantry_obs[np.arange(len(pantry_idxs)), pantry_idxs] = 1
		return pantry_obs

	def gen_shelf_obs(self):
		# here, we may have None's in shelf list, so put -1 for the index there for now
		shelf_idxs = [self.object_to_idx.get(s_type, -1) for s_type in self.shelf_type]
		shelf_obs = np.zeros((len(self.shelf), len(self.object_to_idx) + 1), dtype=np.uint8)
		shelf_obs[np.arange(len(self.shelf)), shelf_idxs] = 1
		# exclude the last column corresponding to Nones
		shelf_obs = shelf_obs[:, :-1]
		return shelf_obs

	def dead(self):
		return super().dead() or any([np.allclose(monster.cur_pos, self.agent_pos) for monster in self.monsters])

	def step(self, action):
		obs, reward, done, info = super().step(action, include_full_img=True)
		obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
		return obs, reward, done, info

	def reset(self):
		obs = super().reset()
		obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
		self.pantry = []
		self.shelf = [None] * 5
		self.shelf_type = [''] * 5
		self.monsters = []
		return obs


class FoodEnvHard6and4(FoodEnvHard):
	def __init__(self):
		super().__init__(health_rate=6)


class FoodEnvHard10and4(FoodEnvHard):
	def __init__(self):
		super().__init__(health_rate=10)


class FoodEnvHard10and4Vision(FoodEnvHard):
	def __init__(self):
		super().__init__(health_rate=10, obs_vision=True)


class FoodEnvHard15and4Vision(FoodEnvHard):
	def __init__(self):
		super().__init__(health_rate=15, obs_vision=True)


register(
	id='MiniGrid-Food-32x32-Hard-4and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvHard'
)

register(
	id='MiniGrid-Food-32x32-Hard-6and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvHard6and4'
)

register(
	id='MiniGrid-Food-32x32-Hard-10and4-v1',
	entry_point='gym_minigrid.envs:FoodEnvHard10and4'
)

register(
	id='MiniGrid-Food-32x32-Hard-10and4-Vision-v1',
	entry_point='gym_minigrid.envs:FoodEnvHard10and4Vision'
)

register(
	id='MiniGrid-Food-32x32-Hard-15and4-Vision-v1',
	entry_point='gym_minigrid.envs:FoodEnvHard15and4Vision'
)
