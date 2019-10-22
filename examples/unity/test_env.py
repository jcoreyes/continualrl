import matplotlib.pyplot as plt
import numpy as np
import sys

from gym_unity.envs import UnityEnv

env_name = "/home/jcoreyes/continual/ml-agents/env_builds/food_collector"  # Name of the Unity environment binary to launch
env = UnityEnv(env_name, worker_id=1, use_visual=False, multiagent=False)
obs = env.reset()
import pdb; pdb.set_trace()

env.close()