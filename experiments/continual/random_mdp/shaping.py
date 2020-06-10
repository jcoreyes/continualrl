import itertools
import numpy as np
import pickle
from datetime import datetime, timedelta

import rlkit.util.hyperparameter as hyp
from utils.markov import gen_mdp, value_iteration, q_learning, MDPEnv, bfs, q_rollout


def shaping_exp(vv):
    results = dict()
    itr_time = orig_time = datetime.now()
    rewards_shaped = np.zeros((vv['n_iters'], vv['max_shape_dist'], vv['shaping_period'], vv['horizon']))
    rewards_unshaped = np.zeros((vv['n_iters'], vv['max_shape_dist'], vv['shaping_period'], vv['horizon']))
    val_rewards_shaped = np.zeros((vv['n_iters'], vv['max_shape_dist'], vv['horizon']))
    val_rewards_unshaped = np.zeros((vv['n_iters'], vv['max_shape_dist'], vv['horizon']))
    mdps = []
    Qs_shaped = np.zeros((vv['n_iters'], vv['s'], vv['a']))
    Qs_unshaped = np.zeros((vv['n_iters'], vv['s'], vv['a']))
    times_shaped = np.zeros((vv['n_iters'], vv['max_shape_dist']))
    times_unshaped = np.zeros((vv['n_iters'], vv['max_shape_dist']))
    for itr in range(vv['n_iters']):
        print(f'Starting iter {itr}...')
        # set goal state to be 0 WLOG
        mdp = gen_mdp(vv['s'], vv['a'], reward=vv['reward'], deterministic=vv['deterministic'], g=vv['g'], sparse=vv['sparse'])
        mdps.append(mdp)
        env = MDPEnv(mdp, vv['horizon'])
        dists = bfs(env.transition, vv['g'])
        max_dist = dists[dists < np.inf].max()
        print(f'Max dist: {max_dist}\nNum states: {np.count_nonzero(dists < np.inf)}')
        max_dist_states = np.where(dists == max_dist)[0]
        val_states = max_dist_states
        Q_shaped = None
        Q_unshaped = None
        for dist in range(1, vv['max_shape_dist']+1):
            print(f'Starting dist {dist}...')
            # shaped
            states = np.nonzero(dists <= dist)[0]
            env.update_reset_distribution(states)
            Q_shaped, rwds_shaped, time_shaped = q_learning(env, vv['shaping_period'], vv['horizon'], Q=Q_shaped, discount=vv['discount'],
                                               eps=vv['eps'], alpha_sched=lambda t: max(0.01, 0.5 - t * 2e-4), return_time=True)
            rewards_shaped[itr, dist-1] = rwds_shaped
            times_shaped[itr, dist-1] = time_shaped
            # unshaped
            env.update_reset_distribution(max_dist_states)
            Q_unshaped, rwds_unshaped, time_unshaped = q_learning(env, vv['shaping_period'], vv['horizon'], Q=Q_unshaped, discount=vv['discount'],
                                                   eps=vv['eps'], alpha_sched=lambda t: max(0.01, 0.5 - t * 2e-4), return_time=True)
            rewards_unshaped[itr, dist-1] = rwds_unshaped
            times_unshaped[itr, dist - 1] = time_unshaped
            # validation for both shaped and unshaped
            env.update_reset_distribution(val_states)
            val_rewards_shaped[itr][dist-1] = q_rollout(env, vv['num_val_rollouts'], vv['horizon'], Q_shaped).mean(axis=0)
            val_rewards_unshaped[itr][dist-1] = q_rollout(env, vv['num_val_rollouts'], vv['horizon'], Q_unshaped).mean(axis=0)
        Qs_shaped[itr] = Q_shaped
        Qs_unshaped[itr] = Q_unshaped
        new_time = datetime.now()
        print(f'Time for iteration {itr}: {(new_time - itr_time).total_seconds()} sec.')
        itr_time = new_time

    results['shaped'] = {
        'train_returns': rewards_shaped,
        'val_returns': val_rewards_shaped,
        'Q': Qs_shaped,
        'time': times_shaped
    }
    results['unshaped'] = {
        'returns': rewards_unshaped,
        'val_returns': val_rewards_unshaped,
        'Q': Qs_unshaped,
        'time': times_unshaped
    }
    results['variant'] = vv
    results['mdps'] = mdps
    pickle.dump(results, open('shaping_%s.pkl' % datetime.now().strftime('%Y_%m_%d_%H_%M_%S'), 'wb'))
    print(f'Total time taken: {(datetime.now() - orig_time).total_seconds()} sec.')


if __name__ == "__main__":
    variant = dict(
        reward='sparse',
        n_iters=5,
        s=1000,
        a=5,
        g=0,
        discount=1,
        horizon=200,
        shaping_period=25,
        # random policy
        eps=1,
        deterministic=True,
        max_shape_dist=150,
        num_val_rollouts=5,
        sparse=True
    )

    shaping_exp(variant)
