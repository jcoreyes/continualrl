from enum import IntEnum

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import MiniGridAbsoluteEnv, Food, GridAbsolute


class FoodEnvBase(MiniGridAbsoluteEnv):
	class Actions(IntEnum):
		# Absolute directions
		west = 0
		east = 1
		north = 2
		south = 3
		mine = 4

	def __init__(self,
				 agent_start_pos=(1, 1),
				 health_cap=5,
				 health_rate=4,
				 food_rate=4,
	             grid_size=8,
				 **kwargs
				 ):
		self.agent_start_pos = agent_start_pos
		# self.agent_start_dir = agent_start_dir
		self.health_rate = health_rate
		self.food_rate = food_rate
		self.health_cap = health_cap
		self.health = health_cap

		super().__init__(
			# Set this to True for maximum speed
			see_through_walls=True,
			grid_size=grid_size,
			**kwargs
		)

		# since superclass constructor overrides actions attribute we put this last
		self.actions = FoodEnvBase.Actions

	def _reward(self):
		return 1

	def _gen_grid(self, width, height):
		# Create an empty grid
		self.grid = GridAbsolute(width, height)

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# Place a goal square in the bottom-right corner
		# self.grid.set(width - 2, height - 2, Goal())

		# Place the agent
		if self.agent_start_pos is not None:
			self.start_pos = self.agent_start_pos
			# self.start_dir = self.agent_start_dir
		else:
			self.place_agent()

		self.mission = None

	def step(self, action):
		done = False
		if not super().step(action, override=True):
			# subclass-defined extra actions. if not caught by that, then unknown action
			if not self.extra_step(action):
				assert False, "unknown action %d" % action

		# decrease health bar
		self.decay_health()
		# generate new food
		self.place_food()

		# generate obs after action is caught and food is placed. generate reward before death check
		obs = self.gen_obs()
		rwd = self._reward()

		# dead.
		if self.health <= 0:
			done = True
			rwd = 0

		return obs, rwd, done, {}

	def extra_step(self, action):
		pass

	def place_food(self):
		pass

	def decay_health(self):
		pass

	def __getstate__(self):
		d = self.__dict__.copy()
		del d['grid_render']
		return d

	def __setstate__(self, d):
		self.__dict__.update(d)
