import random

from rlkit.exploration_strategies.base import RawExplorationStrategy


class EpsilonGreedy(RawExplorationStrategy):
    """
    Take a random discrete action with some probability.
    """

    def __init__(self, action_space, prob_random_action=0.1):
        self.prob_random_action = prob_random_action
        self.action_space = action_space

    def get_action_from_raw_action(self, action, **kwargs):
        if random.random() <= self.prob_random_action:
            return self.action_space.sample()
        return action


class EpsilonGreedySchedule(RawExplorationStrategy):
    """
    Take a random discrete action with some changing probability.
    """

    def __init__(self, action_space, schedule):
        """
        :param schedule: a function that takes one argument - the number of prior calls - and outputs epsilon
        """
        self.action_space = action_space
        self.schedule = schedule
        self.time = 0

    def get_action_from_raw_action(self, action, **kwargs):
        time = self.time
        self.time += 1
        if random.random() <= self.schedule(time):
            return self.action_space.sample()
        return action


class EpsilonGreedyDecay(EpsilonGreedySchedule):
    def __init__(self, action_space, rate, max_eps, min_eps):
        self.rate = rate
        self.max_eps = max_eps
        self.min_eps = min_eps
        super().__init__(action_space, self.schedule)

    def schedule(self, t):
        return max(self.max_eps - self.rate * t, self.min_eps)
