import os
import pickle
import datetime
import numpy as np
from rlkit.envs.swimmer_gather_mujoco.swimmer_gather_env import SwimmerGatherEnv


def gen_validation_envs(n, filename, **kwargs):
    envs = []
    seeds = np.random.randint(0, 100000, n).tolist()
    for idx in range(n):
        env_kwargs = dict(
            ball_radius=0.5,
            radius=100,
            radius_decay=0,
            radius_mode='ball',

            action_noise_mode=None,
            action_noise_std=0,
            action_noise_discount=0.98,
        )
        env_kwargs.update(**kwargs)

        env = SwimmerGatherEnv(
            **env_kwargs
        )
        envs.append(env)
    pickle.dump({'envs': envs, 'seeds': seeds}, open(filename, 'wb'))
    print('Generated %d envs at file: %s' % (n, filename))


if __name__ == '__main__':
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    validation_dir = os.path.join(cur_dir, 'validation_envs')
    os.makedirs(validation_dir, exist_ok=True)

    filename = 'dynamic_static_validation_envs_%s.pkl' % timestamp

    gen_validation_envs(100, os.path.join(validation_dir, filename))
