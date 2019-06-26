from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from rlkit.envs.gym_minigrid.gym_minigrid.envs.getfood_base import FoodEnvBase
from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import CELL_PIXELS, Food


class FoodEnvEasy(FoodEnvBase):
    """
    Pick up food to gain 1 health point,
    Lose 1 health point every `health_rate` timesteps,
    Get 1 reward per timestep
    """

    def __init__(self,
                 init_resources=None,
                 food_rate_decay=0.0,
                 lifespan=0,
                 **kwargs):
        self.init_resources = init_resources or {}
        self.food_rate_decay = food_rate_decay
        self.lifespan = lifespan

        super().__init__(**kwargs)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(12481,) if self.obs_vision else (227,),
            dtype='uint8'
        )

    def extra_step(self, action, matched):
        self.food_rate += self.food_rate_decay

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

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
            for _ in range(count):
                self.place_obj(TYPE_TO_CLASS_ABS[type]())

    def place_items(self):
        self.place_prob(Food(lifespan=self.lifespan), 1 / self.food_rate)


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


class FoodEnvEasy10and4Cap50Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, health_cap=50, food_rate_decay=0.005)


class FoodEnvEasy10and4Cap50Init10Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, health_cap=50, init_resources={'food': 10},
                         food_rate_decay=0.005)


class FoodEnvEasy10and4Cap50Init10DecayVision(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, health_cap=50, init_resources={'food': 10},
                         food_rate_decay=0.005, obs_vision=True)


class FoodEnvEasy10and6Cap50Decay(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, food_rate=6, health_cap=50, food_rate_decay=0.005)


class FoodEnvEasy10and6Cap50DecayLifespan30(FoodEnvEasy):
    def __init__(self):
        super().__init__(health_rate=10, food_rate=6, health_cap=50, food_rate_decay=0.005,
                         lifespan=30)


register(
    id='MiniGrid-Food-8x8-Easy-4and4-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy'
)

register(
    id='MiniGrid-Food-8x8-Easy-6and4-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy6and4'
)

register(
    id='MiniGrid-Food-8x8-Easy-6and4-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy6and4Vision'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and4-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and4'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and4-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and4Vision'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and4-Cap50-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and4Cap50Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and4-Cap50-Init10-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and4Cap50Init10Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and4-Cap50-Init10-Decay-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and4Cap50Init10DecayVision'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and6-Cap50-Decay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and6Cap50Decay'
)

register(
    id='MiniGrid-Food-8x8-Easy-10and6-Cap50-Decay-Lifespan30-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEasy10and6Cap50DecayLifespan30'
)
