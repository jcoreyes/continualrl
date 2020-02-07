import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration, q_learning, MDPEnv


def dynamism_exp(vv):
    orig_time = datetime.now()
    values = np.zeros((vv['n_iters'], len(vv['betas']), vv['s']))
    rewards = np.zeros((vv['n_iters'], len(vv['betas']), vv['num_episodes'], vv['horizon']))
    learn_times = np.zeros((vv['n_iters'], len(vv['betas'])))
    mdps = []
    for itr in range(vv['n_iters']):
        print(f'Running iteration {itr}...')
        clock_time = timedelta(seconds=0)
        mdp = gen_mdp(vv['s'], vv['a'], self_p=vv['self_p'], reward=vv['reward'])
        mdps.append(mdp.copy())
        T = mdp['transition']
        U = np.ones_like(mdp['transition']) / vv['s']

        for idx, beta in enumerate(vv['betas']):
            print(f'Running experiment for beta: {beta}...')

            mdp['transition'] = beta * U + (1 - beta) * T
            env = MDPEnv(mdp, vv['horizon'])

            value_time = datetime.now()
            Q, rwds, learn_time = q_learning(env, vv['num_episodes'], vv['horizon'], discount=vv['discount'],
                                             eps=vv['eps'], alpha_sched=lambda t: max(0.01, 0.5 - t * 2e-4),
                                             return_time=True)
            clock_time += (datetime.now() - value_time)

            values[itr, idx] = Q.max(axis=1)
            rewards[itr, idx] = rwds
            learn_times[itr, idx] = learn_time

        print(f'Iteration {itr} took {clock_time.total_seconds()} sec.')
    results = {'returns': np.mean(rewards, axis=0), 'value': np.mean(values, axis=0), 'variant': vv, 'mdps': mdps}
    pickle.dump(results, open(f'{vv["exp_name"]}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pkl', 'wb'))
    print(f'Total time: ({(datetime.now() - orig_time).total_seconds()} s)')


if __name__ == "__main__":
    variant = dict(
        exp_name='dynamism',
        reward='sparse',
        n_iters=5,
        s=100,
        a=10,
        discount=0.999,
        betas=np.linspace(0, 1, 21, endpoint=True).tolist(),
        mixins=[None, 'uniform'],
        # self-transition probability
        # TODO sweep over self_p's as well
        self_p=0.2,
        num_high=5,
        horizon=200,
        num_episodes=10000,
        eps=0.05
    )

    dynamism_exp(variant)
