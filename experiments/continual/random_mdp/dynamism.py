import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration, q_learning, MDPEnv, q_rollout


def dynamism_exp(vv):
    orig_time = datetime.now()
    values = np.zeros((vv['n_iters'], len(vv['betas']), vv['s']))
    rewards = np.zeros((vv['n_iters'], len(vv['betas']), vv['max_num_episodes'], vv['horizon']))
    val_rewards = np.zeros((vv['n_iters'], len(vv['betas']), vv['horizon']))
    learn_times = np.zeros((vv['n_iters'], len(vv['betas'])))
    mdps = []
    for itr in range(vv['n_iters']):
        print(f'Running iteration {itr}...')
        clock_time = timedelta(seconds=0)
        mdp = gen_mdp(vv['s'], vv['a'], self_p=vv['self_p'], reward=vv['reward'], deterministic=vv['deterministic'], sparse=vv['sparse'], sparse_dist=vv['sparse_dist'])
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
                                             return_time=True, max_num_episodes=vv['max_num_episodes'])
            clock_time += (datetime.now() - value_time)

            mdp['transition'] = T
            env = MDPEnv(mdp, vv['horizon'])
            val_rwds = q_rollout(env, vv['num_val_rollouts'], vv['horizon'], Q).mean(axis=0)

            values[itr, idx] = Q.max(axis=1)
            rewards[itr, idx, :len(rwds)] = rwds
            val_rewards[itr, idx] = val_rwds
            learn_times[itr, idx] = learn_time

        print(f'Iteration {itr} took {clock_time.total_seconds()} sec.')
    results = {
        'rewards': np.mean(rewards, axis=0),
        'val_rewards': np.mean(val_rewards, axis=0),
        'value': np.mean(values, axis=0),
        'variant': vv,
        'mdps': mdps,
        'times': learn_times}
    pickle.dump(results, open(f'{vv["exp_name"]}_{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.pkl', 'wb'))
    print(f'Total time: ({(datetime.now() - orig_time).total_seconds()} s)')


if __name__ == "__main__":
    variant = dict(
        exp_name='dynamism',
        reward='sparse',
        n_iters=1,
        s=1000,
        a=5,
        g=0,
        discount=0.999,
        betas=np.linspace(0, 1, 51, endpoint=True).tolist(),
        self_p=None,
        horizon=200,
        num_episodes=250,
        max_num_episodes=1000,
        eps=1,
        deterministic=False,
        sparse=True,
        sparse_dist=3,
        num_val_rollouts=20
    )

    dynamism_exp(variant)
