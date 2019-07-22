import cv2
from enum import IntEnum
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
from gym import spaces
import numpy as np

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import MiniGridAbsoluteEnv, Food, \
    GridAbsolute, CELL_PIXELS


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
                 health_cap=50,
                 food_rate=4,
                 grid_size=8,
                 obs_vision=False,
                 reward_type='delta',
                 fully_observed=False,
                 can_die=True,
                 **kwargs
                 ):
        self.agent_start_pos = agent_start_pos
        # self.agent_start_dir = agent_start_dir
        self.food_rate = food_rate
        self.health_cap = health_cap
        self.health = health_cap
        self.last_health = self.health
        self.obs_vision = obs_vision
        self.reward_type = reward_type
        self.fully_observed = fully_observed
        self.can_die = can_die
        if not hasattr(self, 'actions'):
            self.actions = FoodEnvBase.Actions
        super().__init__(
            # Set this to True for maximum speed
            see_through_walls=True,
            grid_size=grid_size,
            **kwargs
        )

    def _reward(self):
        if self.reward_type == 'survival':
            rwd = 1
        elif self.reward_type == 'delta':
            rwd = self.health - self.last_health
        elif self.reward_type == 'health':
            rwd = self.health
        else:
            assert False, "Reward type not matched"
        self.last_health = self.health
        return rwd

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

        self.extra_gen_grid()

        self.mission = None

    def step(self, action):
        done = False
        matched = super().step(action, override=True)
        # subclass-defined extra actions. if not caught by that, then unknown action
        if not self.extra_step(action, matched):
            assert False, "unknown action %d" % action

        # decrease health bar
        self.decay_health()
        # generate new food
        self.place_items()
        # generate obs after action is caught and food is placed. generate reward before death check
        img = self.get_img()
        full_img = self.get_full_img(scale=1 if self.fully_observed else 1 / 8)
        rwd = self._reward()

        # tick on each grid item
        to_remove = []
        for j in range(0, self.grid.height):
            for i in range(0, self.grid.width):
                cell = self.grid.get(i, j)
                if cell is not None:
                    if not cell.step():
                        to_remove.append((i, j))
        for idxs in to_remove:
            self.grid.set(*idxs, None)

        # dead.
        if self.dead():
            done = True

        if self.fully_observed:
            obs = np.concatenate(
                (full_img.flatten(), np.array([self.health]), np.array(self.agent_pos)))
        else:
            obs = np.concatenate((img.flatten(), full_img.flatten(), np.array([self.health])))
        return obs, rwd, done, {}

    def reset(self):
        super().reset()
        self.extra_reset()
        img = self.get_img()
        full_img = self.get_full_img()
        if self.fully_observed:
            obs = np.concatenate(
                (full_img.flatten(), np.array([self.health]), np.array(self.agent_pos)))
        else:
            obs = np.concatenate((img.flatten(), full_img.flatten(), np.array([self.health])))
        return obs

    def get_full_img(self, scale=1 / 8):
        """ Return the whole grid view """
        if self.obs_vision:
            full_img = self.get_full_obs_render(scale=scale)
        else:
            full_img = self.grid.encode(self)
        # NOTE: in case need to scale here instead of in above func call: return cv2.resize(
        # full_img, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
        # make the output torch-ready!
        return full_img.transpose(2, 0, 1)

    def get_img(self):
        """ Return the agent view """
        img = self.gen_obs()
        if self.obs_vision:
            img = self.get_obs_render(img, CELL_PIXELS // 4)
        # make the output torch-ready!
        return img.transpose(2, 0, 1)

    def extra_step(self, action, matched):
        return matched

    def extra_reset(self):
        pass

    def place_items(self):
        pass

    def extra_gen_grid(self):
        pass

    def place_prob(self, obj, prob):
        if np.random.binomial(1, prob):
            self.place_obj(obj)

    def decay_health(self):
        self.add_health(-1)

    def add_health(self, num):
        # clip health between 0 and cap after adjustment
        self.health = max(0, min(self.health_cap, self.health + num))

    def dead(self):
        return self.can_die and self.health <= 0

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['grid_render']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class FoodEnvEmptyFullObs(FoodEnvBase):
    def __init__(self):
        super().__init__(fully_observed=True)

    def decay_health(self):
        pass


register(
    id='MiniGrid-Food-8x8-Empty-FullObs-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvEmptyFullObs'
)
