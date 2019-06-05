from gym_minigrid.minigrid_absolute import *
from gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS


class FoodEnv(MiniGridAbsoluteEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""

	def __init__(
		self,
		size=8,
		agent_start_pos=(1,1),
		agent_start_dir=0,
		health_cap=5,
		health_rate=4,
		food_rate=4
	):
		self.agent_start_pos = agent_start_pos
		self.agent_start_dir = agent_start_dir
		self.health_rate = health_rate
		self.food_rate = food_rate
		self.health_cap = health_cap

		super().__init__(
			grid_size=size,
			max_steps=0,
			# Set this to True for maximum speed
			see_through_walls=True
		)

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
			self.start_dir = self.agent_start_dir
		else:
			self.place_agent()

		self.mission = None

	def _reward(self):
		return 1

	def step(self, action):
		obs, rwd, done, info = super().step(action)
		# decrease health bar
		if self.step_count and self.step_count % self.health_rate == 0:
			self.health -= 1
		# generate new food
		if self.step_count % self.food_rate == 0:
			self.place_obj(Food())
		# dead.
		if self.health <= 0:
			done = True
			rwd = 0
		return obs, rwd, done, info


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
