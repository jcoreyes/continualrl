import numpy as np
from gym import Env
from gym.spaces import Discrete


class ProxyEnv(Env):
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()

    def __getattr__(self, attr):
        if attr == '_wrapped_env':
            raise AttributeError()
        return getattr(self._wrapped_env, attr)

    def __getstate__(self):
        """
        This is useful to override in case the wrapped env has some funky
        __getstate__ that doesn't play well with overriding __getattr__.

        The main problematic case is/was gym's EzPickle serialization scheme.
        :return:
        """
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class MultiDiscreteActionEnv(ProxyEnv, Env):
    def __init__(self, wrapped_env, nvec):
        super().__init__(wrapped_env)
        self.nvec = nvec
        self.multi_actions = np.zeros((np.prod(nvec), nvec.shape[0]))
        self.action_space = Discrete(self.multi_actions.shape[0])

        # iterate over all possible combinations of actions by doing addition in different space
        counter = np.zeros(nvec.shape[0])
        for i in range(self.multi_actions.shape[0]):
            while True:
                if (counter >= nvec).sum() == 0:
                    break
                else:
                    idx = (counter == nvec).argmax()
                    counter[idx] = 0
                    counter[idx-1] += 1
            self.multi_actions[i] = counter
            counter[-1] += 1

    def step(self, action):
        multi_action = self.multi_actions[action]
        state, reward, done, info = super().step(multi_action)
        return state, reward, done, {}