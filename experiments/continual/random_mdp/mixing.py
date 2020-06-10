import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration, q_learning, MDPEnv, bfs, q_rollout, gen_transition, gen_init_state, \
    run_mc_stationary


def mixing_exp(vv):
    results = dict()
    orig_time = datetime.now()
    nS = vv['s']
    times = np.full((vv['n_iters'], len(vv['betas'])), np.inf)
    for itr in range(vv['n_iters']):
        print(f'Starting iter {itr}...')
        T = gen_transition(nS, sparse=vv['sparse'], sparse_dist=vv['sparse_dist'], deterministic=vv['deterministic'])
        U = np.ones((nS)) / nS
        start = gen_init_state(nS, num_support=vv['num_support'])
        for idx, beta in enumerate(vv['betas']):
            # print(f'Starting beta {beta}...')
            transition = (1 - beta) * T + beta * U
            time, converged = run_mc_stationary(transition, start, eps=vv['eps'])
            if converged:
                times[itr, idx] = time
    results['times'] = times
    results['variant'] = vv
    pickle.dump(results, open('mixing_%s.pkl' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), 'wb'))
    print(f'Total time taken: {(datetime.now() - orig_time).total_seconds()} sec.')


if __name__ == "__main__":
    variant = dict(
        n_iters=1,
        s=1000,
        num_support=0,
        betas=np.linspace(0, 1, 501, endpoint=True).tolist(),
        sparse=True,
        sparse_dist=3,
        deterministic=False,
        eps=1e-8
    )

    mixing_exp(variant)
