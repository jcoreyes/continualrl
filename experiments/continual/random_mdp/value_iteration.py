import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration


def value_iteration_exp(vv):
    results = dict()
    orig_time = datetime.now()
    for beta, mixin in itertools.product(vv['betas'], vv['mixins']):
        if (beta == 0 and mixin is not None) or (beta != 0 and mixin is None):
            continue
        print(f'Running experiment for beta: {beta}, mixin: {mixin}...')
        clock_time = timedelta(seconds=0)
        times = np.zeros(vv['n_iters'])
        values = np.zeros((vv['n_iters'], vv['s']))
        for itr in range(vv['n_iters']):
            mdp = gen_mdp(vv['s'], vv['a'], self_p=vv['self_p'])
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
            pol, val, time = value_iteration(mdp, vv['eps'], vv['discount'])
            clock_time += (datetime.now() - value_time)
            times[itr] = time
            values[itr] = val
        results[(beta, mixin)] = {
            'time': np.mean(times),
            'value': np.mean(values, axis=0)
        }

        print(f'Value iteration took ({clock_time.total_seconds()} s) for {vv["n_iters"]} runs.')
    results['variant'] = vv
    pickle.dump(results, open('synthetic_mdp_%s.pkl' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), 'wb'))
    print(f'Total time: ({(datetime.now() - orig_time).total_seconds()} s)')


if __name__ == "__main__":
    variant = dict(
        n_iters=20,
        s=100,
        a=10,
        eps=5e-4,
        discount=0.99,
        betas=[0, 0.05, 0.1, 0.15, 0.2],
        mixins=[None, 'uniform', 'shaped'],
        # self-transition probability
        # TODO sweep over self_p's as well
        self_p=0.95,
        num_high=1
    )

    value_iteration_exp(variant)
