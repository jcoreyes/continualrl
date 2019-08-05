from enum import IntEnum

from rlkit.envs.gym_minigrid.gym_minigrid.minigrid_absolute import *
from rlkit.envs.gym_minigrid.gym_minigrid.register import register
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
            health_cap=100,
            food_rate=4,
            max_pantry_size=50,
            obs_vision=False,
            food_rate_decay=0.0,
            init_resources=None,
            her=False,
            skewfit=False,
            **kwargs
    ):
        self.init_resources = init_resources or {}
        self.food_rate_decay = food_rate_decay

        assert not (her and skewfit), "cannot have both HER and skewfit enabled"
        self.her = her
        self.skewfit = skewfit

        # food
        self.pantry = []
        self.max_pantry_size = max_pantry_size
        # other resources
        self.shelf = [None] * 5
        self.shelf_type = [''] * 5
        # stores info for the current timestep
        self.info_last = {}
        self.interactions = {
            ('energy', 'metal'): Axe,
            # edible wood, used for health points
            ('axe', 'tree'): WoodFood
        }
        self.actions = FoodEnvMedium.Actions
        self.object_to_idx = {
            'empty': 0,
            'wall': 1,
            'food': 2,
            'tree': 3,
            'metal': 4,
            'energy': 5,
            'axe': 6,
            'woodfood': 7
        }
        self.food_objs = {obj: idx for obj, idx in self.object_to_idx.items() if obj in FOOD_VALUES}
        self.nonfood_objs = {obj: idx for obj, idx in self.object_to_idx.items() if obj not in FOOD_VALUES}
        super().__init__(
            grid_size=32,
            health_cap=health_cap,
            food_rate=food_rate,
            obs_vision=obs_vision,
            **kwargs
        )

        if self.her:
            # position and inventory obs
            obs_space = spaces.Box(low=0, high=len(self.object_to_idx), shape=(442,), dtype='uint8')
            self.observation_space = spaces.Dict({
                'observation': obs_space,
                'achieved_goal': obs_space,
                'desired_goal': obs_space
            })
        elif self.skewfit:
            obs_space = spaces.Box(low=0, high=len(self.object_to_idx), shape=(442,), dtype='uint8')
            self.observation_space = spaces.Dict({
                'observation': obs_space,
                'state_observation': obs_space,
                'desired_goal': obs_space,
                'state_desired_goal': obs_space,
                'achieved_goal': obs_space,
                'state_achieved_goal': obs_space
            })
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(59001,) if self.obs_vision else (2587,),
                dtype='uint8'
            )

    def place_items(self):
        self.place_prob(Food(), 1 / (self.food_rate + self.step_count * self.food_rate_decay))
        self.place_prob(Metal(), 1 / (2 * self.food_rate))
        self.place_prob(Energy(), 1 / (2 * self.food_rate))
        self.place_prob(Tree(), 1 / (3 * self.food_rate))

    def extra_reset(self):
        if self.her or self.skewfit:
            goal_pos = np.random.randint(1, self.grid_size - 1, size=(2,))

            pantry_size = (self.max_pantry_size, len(self.object_to_idx))
            goal_pantry = np.zeros(pantry_size)
            pantry_idxs_size = np.random.randint(0, self.max_pantry_size // 3)
            pantry_idxs = np.random.choice(list(self.food_objs.values()), size=pantry_idxs_size)
            goal_pantry[np.arange(len(pantry_idxs)), pantry_idxs] = 1

            shelf_size = (len(self.shelf), len(self.object_to_idx))
            shelf_idxs_size = np.random.randint(0, len(shelf_size))
            shelf_idxs = np.random.choice(list(self.nonfood_objs.values()), size=shelf_idxs_size)
            goal_shelf = np.zeros(shelf_size)
            goal_shelf[np.arange(len(shelf_idxs)), shelf_idxs] = 1

            self.goal_obs = np.concatenate((goal_pos, goal_pantry.flatten(), goal_shelf.flatten()))

            print('goal pantry', goal_pantry.sum(axis=0))
            print('goal shelf', goal_shelf.sum(axis=0))

    def extra_gen_grid(self):
        for type, count in self.init_resources.items():
            for _ in range(count):
                self.place_obj(TYPE_TO_CLASS_ABS[type]())

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
                    # uncomment to add mined objs to info
                    # self.info_last.update({agent_cell.type: 1})
                    self.grid.set(*self.agent_pos, None)

        # Consume stored food.
        elif action == self.actions.eat:
            self.pantry.sort(key=lambda item: item.food_value(), reverse=True)
            if self.pantry:
                eaten = self.pantry.pop(0)
                if Axe().type in self.shelf_type or Wood().type in self.shelf_type:
                    self.add_health(eaten.food_value() * 2)
                else:
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

    def step(self, action):
        obs, reward, done, info = super().step(action)
        obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
        if self.her:
            obs = self.gen_her_obs()
        elif self.skewfit:
            obs = self.gen_skewfit_obs()
        if self.her or self.skewfit:
            reward = self.compute_reward(obs['achieved_goal'], self.goal_obs, info)

        info.update(self.info_last)
        self.info_last = {}
        return obs, reward, done, info

    def gen_her_obs(self):
        achieved_goal = np.concatenate(
            (self.agent_pos, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
        obs = {'observation': achieved_goal, 'desired_goal': self.goal_obs, 'achieved_goal': achieved_goal}
        return obs

    def gen_skewfit_obs(self):
        achieved_goal = np.concatenate(
            (self.agent_pos, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
        obs = {
            'observation': achieved_goal,
            'state_observation': achieved_goal,
            'desired_goal': self.goal_obs,
            'state_desired_goal': self.goal_obs,
            'achieved_goal': achieved_goal,
            'state_achieved_goal': achieved_goal
        }
        return obs

    def compute_reward(self, achieved_goal, desired_goal, info):
        assert self.her or self.skewfit, "`compute_reward` function should only be used for HER or skewfit"

        pos, pantry, shelf = np.split(achieved_goal, (2, 402))
        pantry = pantry.reshape((self.max_pantry_size, -1))
        shelf = shelf.reshape((len(self.shelf), -1))

        goal_pos, goal_pantry, goal_shelf = np.split(desired_goal, (2, 402))
        goal_pantry = goal_pantry.reshape((self.max_pantry_size, -1))
        goal_shelf = goal_shelf.reshape((len(self.shelf), -1))

        pantry_error = np.linalg.norm(goal_pantry.sum(axis=0) - pantry.sum(axis=0), ord=1)
        shelf_error = np.linalg.norm(goal_shelf.sum(axis=0) - shelf.sum(axis=0), ord=1)
        pos_error = np.linalg.norm(goal_pos - pos, ord=1)
        # 0.25 factor to match scale of pantry+shelf error
        error = 0.25 * pos_error + pantry_error + shelf_error

        return -error

    def reset(self):
        obs = super().reset()
        obs = np.concatenate((obs, self.gen_pantry_obs().flatten(), self.gen_shelf_obs().flatten()))
        self.pantry = []
        self.shelf = [None] * 5
        self.shelf_type = [''] * 5

        if self.her:
            return self.gen_her_obs()
        elif self.skewfit:
            return self.gen_skewfit_obs()
        else:
            return obs


class FoodEnvMediumCap100(FoodEnvMedium):
    pass


class FoodEnvMediumCap150(FoodEnvMedium):
    def __init__(self):
        super().__init__(health_cap=150)


class FoodEnvMediumCap150Vision(FoodEnvMedium):
    def __init__(self):
        super().__init__(health_cap=150, obs_vision=True)


class FoodEnvMediumCap100InitDecay(FoodEnvMedium):
    def __init__(self):
        super().__init__(health_cap=100, food_rate_decay=0.01,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMediumCap100InitDecayHER(FoodEnvMedium):
    def __init__(self):
        super().__init__(health_cap=100, food_rate_decay=0.01, her=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


class FoodEnvMediumCap100InitDecaySkewFit(FoodEnvMedium):
    def __init__(self):
        super().__init__(health_cap=100, food_rate_decay=0.01, skewfit=True,
                         init_resources={
                             'axe': 8,
                             'woodfood': 5,
                             'food': 15
                         })


register(
    id='MiniGrid-Food-32x32-Medium-Cap100-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap100'
)

register(
    id='MiniGrid-Food-32x32-Medium-Cap150-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap150'
)

register(
    id='MiniGrid-Food-32x32-Medium-Cap150-Vision-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap150Vision'
)

register(
    id='MiniGrid-Food-32x32-Medium-Cap100-InitDecay-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap100InitDecay'
)

register(
    id='MiniGrid-Food-32x32-Medium-Cap100-InitDecay-HER-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap100InitDecayHER'
)

register(
    id='MiniGrid-Food-32x32-Medium-Cap100-InitDecay-SkewFit-v1',
    entry_point='rlkit.envs.gym_minigrid.gym_minigrid.envs:FoodEnvMediumCap100InitDecaySkewFit'
)
