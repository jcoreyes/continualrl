from rlkit.policies.base import Policy
from rlkit.torch.core import torch_ify, eval_np, np_ify
from torch import nn
import numpy as np
from torch.distributions import Categorical


class GoalSetterNetwork(Policy, nn.Module):
    def __init__(self, policy, horizon):
        super().__init__()
        self.policy = policy
        # period of time between setting goals
        self.horizon = horizon
        self.count = 0
        # below two variables used for goal transitions
        self.goal = None
        self.last_obs = None

    def forward(self, obs):
        ac, _ = self.get_action(np_ify(obs))
        return torch_ify(ac)

    def get_action(self, obs_np):
        if self.count == 0:
            reward = 0
        else:
            reward = self.get_reward(obs_np, self.goal, None, None)
        self.goal, new_goal = self.peek_action(obs_np)
        self.last_obs = obs_np
        self.count += 1
        return self.goal, {'new_goal': new_goal, 'reward': reward}

    def peek_action(self, obs_np):
        if self.count % self.horizon == 0:
            goal, _ = self.policy.get_action(obs_np)
            new_goal = True
        else:
            goal = self.goal_transition(self.last_obs, self.goal, obs_np)
            new_goal = False
        return goal, new_goal

    @staticmethod
    def goal_transition(obs, goal, new_obs):
        return goal + obs - new_obs

    def get_reward(self, obs, goal, ac, next_obs):
        return -np.linalg.norm(obs - goal, ord=2)


class GoalSetterWrappedWithExplorationStrategy(Policy, nn.Module):
    def __init__(self, exploration_strategy, setter):
        super().__init__()
        self.es = exploration_strategy
        self.setter = setter
        self.t = 0

    def set_num_steps_total(self, t):
        self.t = t

    def forward(self, *args, **kwargs):
        ac, _ = self.get_action(*args, **kwargs)
        return torch_ify(ac)

    def get_action(self, *args, **kwargs):
        return self.es.get_action(self.t, self.setter, *args, **kwargs)

    def reset(self):
        self.es.reset()
        self.setter.reset()

    def peek_action(self, *args, **kwargs):
        return self.setter.peek_action(*args, **kwargs)

    def goal_transition(self, *args, **kwargs):
        return self.setter.goal_transition(*args, **kwargs)

    def get_reward(self, *args, **kwargs):
        return self.setter.get_reward(*args, **kwargs)
