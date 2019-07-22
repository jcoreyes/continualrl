from enum import IntEnum

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, MiniGridAbsoluteEnv, DIR_TO_VEC, \
	GridAbsolute
from rlkit.torch.core import torch_ify
from rlkit.torch.networks import Mlp
from torch.optim import Adam
from torch.nn import MSELoss


class FoodEnvMedium1Inv(FoodEnvBase):
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
		place = 6

	def __init__(
			self,
			grid_size=32,
			health_cap=100,
			food_rate=4,
			max_pantry_size=50,
			obs_vision=False,
			food_rate_decay=0.0,
			init_resources=None,
			gen_resources=True,
			make_rtype='sparse',
			lifespan=0,
			task=None,
			rnd=False,
			cbe=False,
			woodfood=True,
			**kwargs
	):
		self.init_resources = init_resources or {}
		self.food_rate_decay = food_rate_decay
		self.lifespan = lifespan
		self.interactions = {
			('energy', 'metal'): 'axe',
			# edible wood, used for health points
			('axe', 'tree'): 'woodfood' if woodfood else 'wood',
		}
		self.ingredients = {v: k for k, v in self.interactions.items()}
		self.gen_resources = gen_resources

		# TASK stuff
		self.task = task.split()  # e.g. 'pickup axe', 'navigate 3 5', 'make wood'
		self.make_sequence = self.get_make_sequence()
		# print(self.make_sequence)
		self.make_rtype = make_rtype
		self.max_make_idx = -1
		self.made_obj_type = None
		self.last_placed_on = None

		# Exploration!
		assert not (cbe and rnd), "can't have both CBE and RND"
		# CBE
		self.cbe = cbe
		# RND
		self.rnd = rnd
		self.obs_count = {}
		# below two variables are to keep running count of stdev for RND normalization
		self.sum_rnd = 0
		self.sum_square_rnd = 0
		self.rnd_loss = MSELoss()

		# food
		self.pantry = []
		self.max_pantry_size = max_pantry_size
		# stores info for the current timestep
		self.info_last = {}
		self.actions = FoodEnvMedium1Inv.Actions
		self.object_to_idx = {
			'empty': 0,
			'wall': 1,
			'food': 2,
			'tree': 3,
			'metal': 4,
			'energy': 5,
			'axe': 6,
		}
		if woodfood:
			self.object_to_idx.update({'woodfood': 7})
		else:
			self.object_to_idx.update({'wood': 7})

		super().__init__(
			grid_size=grid_size,
			health_cap=health_cap,
			food_rate=food_rate,
			obs_vision=obs_vision,
			**kwargs
		)

		shape = None
		if self.grid_size == 32:
			if self.obs_vision:
				shape = (58969,)
			else:
				if self.fully_observed:
					shape = (2459,)
				else:
					shape = (2555,)
		elif self.grid_size == 16:
			if not self.obs_vision and self.fully_observed:
				shape = (923,)

		if shape is None:
			raise TypeError("Env configuration not supported")

		self.observation_space = spaces.Box(
			low=0,
			high=255,
			shape=shape,
			dtype='uint8'
		)

		if self.rnd:
			self.rnd_network = Mlp([128, 128], 32, self.observation_space.low.size)
			self.rnd_target_network = Mlp([128, 128], 32, self.observation_space.low.size)
			self.rnd_optimizer = Adam(self.rnd_target_network.parameters(), lr=3e-4)

	def get_make_sequence(self):
		def add_ingredients(obj, seq):
			if obj in self.ingredients:
				for ingredient in self.ingredients[obj]:
					add_ingredients(ingredient, seq)
				seq.append(obj)
			else:
				seq.append(obj)
		make_sequence = []
		goal_obj = self.task[1]
		add_ingredients(goal_obj, make_sequence)
		return make_sequence

	def place_items(self):
		if self.gen_resources:
			self.place_prob(Food(lifespan=self.lifespan), 1 / (self.food_rate + self.step_count * self.food_rate_decay))
			self.place_prob(Metal(lifespan=self.lifespan), 1 / (2 * self.food_rate))
			self.place_prob(Energy(lifespan=self.lifespan), 1 / (2 * self.food_rate))
			self.place_prob(Tree(lifespan=self.lifespan), 1 / (3 * self.food_rate))

	def extra_gen_grid(self):
		for type, count in self.init_resources.items():
			if self.task and self.task[0] == 'pickup' and type == self.task[1]:
				for _ in range(count):
					self.place_obj(TYPE_TO_CLASS_ABS[type]())
			else:
				for _ in range(count):
					self.place_obj(TYPE_TO_CLASS_ABS[type](lifespan=self.lifespan))

	def extra_step(self, action, matched):
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
					self.info_last.update({agent_cell.type: 1})
					self.grid.set(*self.agent_pos, None)

		# Consume stored food.
		elif action == self.actions.eat:
			self.pantry.sort(key=lambda item: item.food_value(), reverse=True)
			if self.pantry:
				eaten = self.pantry.pop(0)
				if self.carrying and self.carrying.type == Axe().type:
					self.add_health(eaten.food_value() * 2)
				else:
					self.add_health(eaten.food_value())

		# actions to use each element of inventory
		elif action == self.actions.place:
			self.place_act()

		else:
			matched = False

		return matched

	def place_act(self):
		agent_cell = self.grid.get(*self.agent_pos)
		if self.carrying is None:
			# there's nothing to place
			return
		elif agent_cell is None:
			# there's nothing to combine it with, so just place it on the grid
			self.grid.set(*self.agent_pos, self.carrying)
		else:
			# let's try to combine the placed object with the existing object
			interact_tup = tuple(sorted([self.carrying.type, agent_cell.type]))
			new_type = self.interactions.get(interact_tup, None)
			if not new_type:
				# the objects cannot be combined, no-op
				return
			else:
				self.last_placed_on = agent_cell
				# replace existing obj with new obj
				new_obj = TYPE_TO_CLASS_ABS[new_type]()
				self.grid.set(*self.agent_pos, new_obj)
				self.made_obj_type = new_obj.type
		# remove placed object from inventory
		self.carrying = None

	def add_to_shelf(self, obj):
		""" Returns whether adding to shelf succeeded """
		if self.carrying is None:
			self.carrying = obj
			return True
		return False

	def gen_pantry_obs(self):
		pantry_obs = np.zeros((self.max_pantry_size, len(self.object_to_idx)), dtype=np.uint8)
		pantry_idxs = [self.object_to_idx[obj.type] for obj in self.pantry]
		pantry_obs[np.arange(len(pantry_idxs)), pantry_idxs] = 1
		return pantry_obs

	def gen_shelf_obs(self):
		""" Return one-hot encoding of carried object type. """
		shelf_obs = np.zeros((1, len(self.object_to_idx)), dtype=np.uint8)
		if self.carrying is not None:
			shelf_obs[0, self.object_to_idx[self.carrying.type]] = 1
		return shelf_obs

	def step(self, action):
		obs, reward, done, info = super().step(action)
		pantry_obs = self.gen_pantry_obs()
		shelf_obs = self.gen_shelf_obs()

		extra_obs = np.concatenate((pantry_obs.flatten(), shelf_obs.flatten()))
		extra_obs_count_string = np.concatenate((pantry_obs.sum(axis=0), shelf_obs.sum(axis=0))).tostring()
		obs = np.concatenate((obs, extra_obs))
		solved = self.solved_task()
		if self.task and self.task[0] == 'make':
			reward = self.get_make_reward()
			info.update({'progress': (self.max_make_idx + 1) / len(self.make_sequence)})
		else:
			reward = int(solved)

		if solved:
			done = True
			info.update({'solved': True})
		else:
			info.update({'solved': False})
		# Exploration bonuses
		if self.cbe:
			self.obs_count[extra_obs_count_string] = self.obs_count.get(extra_obs_count_string, 0) + 1
			reward += 1 / np.sqrt(self.obs_count[extra_obs_count_string])
		elif self.rnd:
			torch_obs = torch_ify(extra_obs)
			true_rnd = self.rnd_network(torch_obs)
			pred_rnd = self.rnd_target_network(torch_obs)
			loss = self.rnd_loss(true_rnd, pred_rnd)

			self.rnd_optimizer.zero_grad()
			loss.backward()
			self.rnd_optimizer.step()
			# RND exploration bonus
			self.sum_rnd += loss
			self.sum_square_rnd += loss ** 2
			stdev = (self.sum_square_rnd / self.step_count) - (self.sum_rnd / self.step_count) ** 2
			reward += loss / (stdev * self.health_cap)

		return obs, reward, done, info

	def reset(self):
		obs = super().reset()
		obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
		self.pantry = []
		self.made_obj_type = None
		self.last_placed_on = None
		self.max_make_idx = -1
		return obs

	def solved_task(self):
		if self.task:
			if self.task[0] == 'navigate':
				pos = np.array(self.task[1:])
				return np.array_equal(pos, self.agent_pos)
			elif self.task[0] == 'pickup':
				return self.carrying is not None and (self.carrying.type == self.task[1])
			elif self.task[0] == 'make':
				return self.carrying is not None and self.carrying.type == self.task[1]
		return False

	def get_make_reward(self):
		reward = 0
		if self.make_rtype == 'sparse':
			reward = int(self.solved_task())
		elif self.make_rtype in ['dense', 'waypoint']:
			carry_idx = self.make_sequence.index(self.carrying.type) if self.carrying and self.carrying.type in self.make_sequence else -1
			place_idx = self.make_sequence.index(self.last_placed_on.type) if self.last_placed_on and self.last_placed_on.type in self.make_sequence else -1
			idx = max(carry_idx, place_idx)
			true_idx = min(idx, self.max_make_idx)
			if idx == self.max_make_idx + 1:
				reward = 50
				self.max_make_idx = idx
				if idx == len(self.make_sequence) - 1:
					reward = 100
			elif self.make_rtype == 'dense':
				next_pos = self.get_closest_obj_pos(self.make_sequence[true_idx + 1])
				if next_pos is not None:
					dist = np.linalg.norm(next_pos - self.agent_pos, ord=1)
					reward = -dist
				# else there is no obj of that type, so 0 reward
		else:
			raise TypeError('Make reward type "%s" not recognized' % self.make_rtype)
		return reward

	def get_closest_obj_pos(self, type=None):
		def test_point(point, type):
			try:
				obj = self.grid.get(*point)
				if obj and (type is None or obj.type == type):
					return True
			except AssertionError:
				# OOB grid access
				return False
		corner = np.array([0, -1])
		for i in range(0, self.grid_size-2):
			# width of the fixed distance level set (diamond shape centered at agent pos)
			width = i + 1
			test_pos = self.agent_pos + corner * i
			for j in range(width):
				if test_point(test_pos, type):
					return test_pos
				test_pos += np.array([1, 1])
			test_pos -= np.array([1, 1])
			for j in range(width):
				if test_point(test_pos, type):
					return test_pos
				test_pos += np.array([-1, 1])
			test_pos -= np.array([-1, 1])
			for j in range(width):
				if test_point(test_pos, type):
					return test_pos
				test_pos += np.array([-1, -1])
			test_pos -= np.array([-1, -1])
			for j in range(width):
				if test_point(test_pos, type):
					return test_pos
				test_pos += np.array([1, -1])
		return None


class FoodEnvMedium1InvCap50(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=50)


class FoodEnvMedium1InvCap100(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=100)


class FoodEnvMedium1InvCap100Vision(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=10, obs_vision=True)


class FoodEnvMedium1InvCap500InitDecay(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=5, food_rate_decay=0.01,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15
						 })


class FoodEnvMedium1InvCap500InitDecayLifespan80(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=500, food_rate_decay=0.01, lifespan=80,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15
						 })


class FoodEnvMedium1InvCap500InitDecayFullObs(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=500, food_rate_decay=0.01, fully_observed=True,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15
						 })


class FoodEnvMedium1InvCap500InitDecayLifespan80FullObs(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=500, food_rate_decay=0.01, lifespan=80, fully_observed=True,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15
						 })


class FoodEnvMedium1InvCap2500InitDecayLifespan200FullObs(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=2500, food_rate_decay=0.01, lifespan=200, fully_observed=True,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15
						 })


class FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200Task(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, food_rate_decay=0.01, lifespan=200, fully_observed=True, task='make axe',
		                 make_rtype='dense',
						 init_resources={
							 'axe': 8,
							 # 'woodfood': 5,
							 'food': 15,
							 'metal': 30,
							 'energy': 30
						 })


class FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200TaskCBE(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, food_rate_decay=0.01, lifespan=200, fully_observed=True, task='pickup axe',
		                 cbe=True,
						 init_resources={
							 'axe': 8,
							 'woodfood': 5,
							 'food': 15,
						 })


class FoodEnvMedium1Inv1TierDenseReward16(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(grid_size=16, health_cap=1000, gen_resources=False, fully_observed=True, task='make energy',
		                 make_rtype='dense',
						 init_resources={
							 'energy': 4
						 })


class FoodEnvMedium1Inv2TierDenseReward(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
		                 make_rtype='dense',
						 init_resources={
							 # 'food': 6,
							 'metal': 12,
							 'energy': 12
						 })


class FoodEnvMedium1Inv2TierDenseReward16(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(grid_size=16, health_cap=1000, gen_resources=False, fully_observed=True, task='make axe',
		                 make_rtype='dense',
						 init_resources={
							 # 'food': 6,
							 'metal': 12,
							 'energy': 12
						 })


class FoodEnvMedium1Inv2TierWaypointReward(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make axe',
		                 make_rtype='waypoint',
						 init_resources={
							 'food': 15,
							 'metal': 30,
							 'energy': 30
						 })


class FoodEnvMedium1Inv3TierDenseReward(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make wood', woodfood=False,
		                 make_rtype='dense',
						 init_resources={
							 'food': 15,
							 'metal': 30,
							 'energy': 30,
							 'tree': 15
						 })


class FoodEnvMedium1Inv3TierWaypointReward(FoodEnvMedium1Inv):
	def __init__(self):
		super().__init__(health_cap=1000, lifespan=200, fully_observed=True, task='make wood', woodfood=False,
		                 make_rtype='waypoint',
						 init_resources={
							 'food': 15,
							 'metal': 30,
							 'energy': 30,
							 'tree': 15
						 })


register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap50-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap50'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap100'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-Vision-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap100Vision'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecay'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-Lifespan80-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayLifespan80'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap100-Init-Decay-FullObs-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayFullObs'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-5and4-Cap100-Init-Decay-FullObs-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv5and4Cap100InitDecayFullObs'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap500-Init-Decay-Lifespan80-FullObs-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap500InitDecayLifespan80FullObs'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap2500-Init-Decay-Lifespan200-FullObs-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap2500InitDecayLifespan200FullObs'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap1000-Init-Decay-FullObs-Lifespan200-Task-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200Task'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-Cap1000-Init-Decay-FullObs-Lifespan200-Task-CBE-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1InvCap1000InitDecayFullObsLifespan200TaskCBE'
)

register(
	id='MiniGrid-Food-16x16-Medium-1Inv-1Tier-Dense-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv1TierDenseReward16'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-2Tier-Dense-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseReward'
)

register(
	id='MiniGrid-Food-16x16-Medium-1Inv-2Tier-Dense-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierDenseReward16'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-2Tier-Waypoint-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv2TierWaypointReward'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-3Tier-Dense-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierDenseReward'
)

register(
	id='MiniGrid-Food-32x32-Medium-1Inv-3Tier-Waypoint-v1',
	entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMedium1Inv3TierWaypointReward'
)
