import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration, q_learning, MDPEnv


def q_learning_exp(vv):
    results = dict()
    orig_time = datetime.now()
    for beta, mixin in itertools.product(vv['betas'], vv['mixins']):
        if (beta == 0 and mixin is not None) or (beta != 0 and mixin is None):
            continue
        print(f'Running experiment for beta: {beta}, mixin: {mixin}...')
        clock_time = timedelta(seconds=0)
        values = np.zeros((vv['n_iters'], vv['s']))
        rewards = np.zeros((vv['n_iters'], vv['num_episodes'], vv['horizon']))
        for itr in range(vv['n_iters']):
            mdp = gen_mdp(vv['s'], vv['a'], self_p=vv['self_p'], reward=vv['reward'])
            env = MDPEnv(mdp, vv['horizon'])
            if beta != 0:
                if mixin == 'uniform':
                    U = np.ones_like(mdp['transition']) / vv['s']  # uniform transition probabilities
                    # bias towards uniform
                    mdp['transition'] = beta * U + (1 - beta) * mdp['transition']
                elif mixin == 'shaped':
                    _, val, _ = value_iteration(mdp, vv['eps'], vv['discount'])
                    S = np.zeros_like(mdp['transition'])
                    # the most shaped transition would have you always reach the highest value state
                    S[:, :, val.argmax()] = 1
                    mdp['transition'] = beta * S + (1 - beta) * mdp['transition']
            value_time = datetime.now()
            Q, rwds = q_learning(env, vv['num_episodes'], vv['horizon'], discount=vv['discount'], eps=vv['eps'], alpha_sched=lambda t: max(0.01, 0.5 - t * 2e-4))
            clock_time += (datetime.now() - value_time)
            values[itr] = Q.max(axis=1)
            rewards[itr] = rwds
        results[(beta, mixin)] = {
            'returns': np.mean(rewards, axis=(0, 1)),
            'value': np.mean(values, axis=0)
        }

        print(f'Q learning took ({clock_time.total_seconds()} s) for {vv["n_iters"]} runs.')
    results['variant'] = vv
    pickle.dump(results, open('q_learning_%s.pkl' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), 'wb'))
    print(f'Total time: ({(datetime.now() - orig_time).total_seconds()} s)')


if __name__ == "__main__":
    variant = dict(
        reward='distance',
        n_iters=20,
        s=100,
        a=10,
        discount=0.999,
        betas=np.linspace(0, 1, 21, endpoint=True).tolist(),
        mixins=[None, 'uniform', 'shaped'],
        # self-transition probability
        # TODO sweep over self_p's as well
        self_p=0.2,
        num_high=5,
        horizon=500,
        num_episodes=100,
        eps=0.05
    )

    q_learning_exp(variant)
