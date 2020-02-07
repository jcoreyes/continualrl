from queue import Queue
import random
import sys
import numpy as np


def gen_transition(s, self_p=None, deterministic=False, sparse=False, sparse_dist=7):
    """
    :param s: int, number of states
    :param self_p: float, if specified gives the self transition probability
    :param deterministic: bool, whether or not the transition probabilities are deterministic. mutually exclusive with self_p
    :return: random transition matrix, with elements sampled uniformly [0, 1] and then normalized by row
    """
    if deterministic:
        T = np.zeros((s, s))
        # each state transitions to a random state with probability 1
        if sparse:
            for i in range(s):
                rand_state = np.random.randint(i-sparse_dist, i+sparse_dist+1)
                # wrap so that anything greater than s becomes
                if rand_state >= s:
                    rand_state -= s
                T[i, rand_state] = 1
        else:
            T[np.arange(s), np.random.randint(0, s, s)] = 1
    else:
        if sparse:
            T = np.zeros((s, s))
            for i in range(s):
                T[i, i: min(s, i+sparse_dist+1)] = np.random.uniform(0, 1, min(s, i+sparse_dist+1)-i)
                if i + sparse_dist + 1 - s > 0:
                    T[i, 0: i+sparse_dist+1-s] = np.random.uniform(0, 1, i+sparse_dist+1-s)
                T[i, max(0, i-sparse_dist):i] = np.random.uniform(0, 1, i-max(0, i-sparse_dist))
                if i - sparse_dist < 0:
                    T[i, i-sparse_dist:] = np.random.uniform(0, 1, sparse_dist-i)
        else:
            T = np.random.uniform(0, 1, (s, s))

        if self_p is not None:
            for i in range(s):
                T[i, i] = self_p / (1 - self_p) * (T[i].sum() - T[i, i])
        # normalize so rows sum to 1
        T /= T.sum(axis=1, keepdims=True)
    return T


def gen_init_state(s, num_support=0):
    """
    :param s: int, number of states
    :param num_support: int, number of states the distribution is supported on
    :return: categorical initial probability distribution over S states
    """
    if num_support:
        v = np.zeros(s)
        v[np.random.choice(np.arange(s), num_support, replace=False)] = np.random.uniform(0, 1, num_support)
    else:
        v = np.random.uniform(0, 1, s)
    # normalize so sums to 1
    return v / v.sum()


def gen_reward(s, a, num_high=0):
    """
    :param s: int, number of states
    :param a: int, number of actions
    :param num_high: int, number of states to randomly set to be high in value
    :return: random reward function
    """
    assert num_high <= s, 'number of high-value states cannot be higher than total number of states'
    R = np.random.uniform(-1, 1, (s, a))
    R[0] = 10
    # for high_state in sample(range(s), num_high):
    #     R[high_state] = 5
    return R


def gen_reward_sparse(s, a, T, g=None):
    """
    Reward of -1 unless at goal
    :param s: int, number of states
    :param a: int, number of actions
    :param T: transition matrix (to give reward for actions that lead to goal)
    :param g: int, 0-index of goal state
    :return: reward function
    """
    if g is None:
        g = 0
    else:
        assert 0 <= g < s, 'goal state must be in [0, s)'
    R = np.zeros((s, a))
    # Any action that gets you to the goal gets you reward of 1. Assumes deterministic actions
    R[np.where(T[:, :, g] != 0)] = 1
    return R


def gen_reward_dist(s, a, g, T):
    # TODO this still rewards based on S instead of S'
    dists = bfs(T, g)
    R = -np.broadcast_to(dists[:, np.newaxis], (s, a))
    return R


def gen_mdp(s, a, self_p=None, num_high=0, g=None, reward='distance', deterministic=False, sparse=False, sparse_dist=7):
    """
    :param s: int, number of states
    :param a: int, number of actions
    :param self_p: float, if specified gives the self transition probability
    :param num_high: int, number of states to set as high-value (used for random reward)
    :param g: int, the goal state (used for distance reward)
    :param reward: string, type of reward {'distance', 'random', 'constant}
    :param deterministic: bool, whether or not the transition probabilities are deterministic. mutually exclusive with self_p
    :param sparse: bool, whether transition matrix is sparse
    :return: dict,
            'state': State space is np.arange(s)
            'action': Action space is np.arange(a)
            'transition': Transition matrix is indexed by cur_state, action, next_state
            'reward': Reward function is indexed by state action
    """
    S = np.arange(s)
    A = np.arange(a)
    T = np.stack([gen_transition(s, self_p=self_p, deterministic=deterministic, sparse=sparse, sparse_dist=sparse_dist) for _ in range(a)])
    # T[0] = np.eye(s)  # add no-op action
    T = np.swapaxes(T, 0, 1)  # (S, A, S')
    if reward == 'sparse':
        R = gen_reward_sparse(s, a, T, g=g)
    elif reward == 'random':
        R = gen_reward(s, a, num_high=num_high)
    elif reward == 'distance':
        R = gen_reward_dist(s, a, g, T)
    else:
        raise ValueError(f'Reward type {reward} not recognized.')
    return {
        'state': S,
        'action': A,
        'transition': T,
        'reward': R
    }


class MDPEnv:
    def __init__(self, mdp, horizon):
        assert horizon > 0
        self.state = mdp['state']
        self.action = mdp['action']
        self.transition = mdp['transition']
        self.reward = mdp['reward']
        self.horizon = horizon
        self.obs = None
        self.t = 0
        self.dist = None

    def reset(self):
        self.obs = np.random.choice(self.state if self.dist is None else self.dist)
        self.t = 0
        return self.obs

    def step(self, action):
        self.t += 1
        reward = self.reward[self.obs, action]
        self.obs = np.random.choice(self.state, p=self.transition[self.obs, action])
        done = self.t >= self.horizon
        return self.obs, reward, done

    def update_reset_distribution(self, dist):
        self.dist = dist


def value_iteration(env, eps=1e-4, discount=0.99):
    """
    Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Value%20Iteration%20Solution.ipynb
    :param env: dict, keys {'state', 'action', 'transition', 'reward'} as returned by `gen_mdp` above
    :param eps: stopping condition for convergence is ||V_{t+1}(s)-V_t(s)||_1 < eps
    :param discount
    :return: deterministic policy, value function, num iterations taken
    """
    assert eps >= 0
    # assert discount < 1, 'only discount < 1 supported, finite horizon not implemented'
    assert all(k in env for k in ('state', 'action', 'transition', 'reward')),\
        'env must contain keys for "state", "action", "transition", "reward"'

    nA = env['action'].shape[0]
    nS = env['state'].shape[0]

    def one_step_lookahead(V):
        """
        Computes Q(STATE, ACTION) over all state-action pairs
        :param V: 1-dim np.ndarray, value function over states
        :return: 2-dim np.ndarray, one-step lookahead value of each state-action pair
        """
        # reward is independent of next state
        reward = np.broadcast_to(env['reward'][..., np.newaxis], (nS, nA, nS))
        # one-step value estimate of taking action A from state S to end up in state S', indexed by (S, A, S')
        one_step_values = reward + discount * V  # V gets broadcasted to the last dimension (S') as desired
        # Q(S, A) = E_{S'~T(S, A)}[R(S,A) + gamma * V(S')], expected value of one_step_values under transition probs
        Q = (env['transition'] * one_step_values).sum(axis=-1)
        # Q is indexed by (S, A)
        return Q

    V = np.zeros(nS)
    Q = np.zeros((nS, nA))
    delta = eps
    num_iters = 0
    while delta >= eps:
        Q = one_step_lookahead(V)
        V_new = Q.max(axis=1)
        # TV(P, Q) on finite set is 0.5 * ||P-Q||_1
        delta = 0.5 * np.linalg.norm(V_new - V, ord=1)
        V = V_new
        num_iters += 1

    policy = np.zeros((nS, nA))
    policy[np.arange(nS), Q.argmax(axis=1)] = 1.0

    return policy, V, num_iters


def eps_greedy_policy(Q, eps=0.05):
    nA = Q.shape[1]

    def policy(obs):
        """ Returns categorical distribution over actions """
        A = np.ones(nA) * eps / nA
        A[Q[obs].argmax()] += 1 - eps
        return A
    return policy


def deterministic_policy(Q):
    return lambda obs: Q[obs].argmax()


def q_rollout(env, num_episodes, horizon, Q):
    rewards = np.zeros((num_episodes, horizon))
    actions = np.zeros((num_episodes, horizon))
    policy = deterministic_policy(Q)
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        t = 0
        while not done and t < horizon:
            action = policy(obs)
            next_obs, reward, done = env.step(action)
            # collect reward
            rewards[ep, t] = reward
            actions[ep, t] = action
            # time update
            obs = next_obs
            t += 1
    return rewards


def q_learning(env, num_episodes, horizon, Q=None, discount=0.99, alpha_sched=lambda t: 0.01, eps=0.05, return_time=False):
    """
    Adapted from https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb
    :param env: MDPEnv object
    :param num_episodes: how many episodes to train for
    :param horizon: how many timesteps per episode
    :param Q: np.ndarray, (optional) initial Q values of shape (ENV.state, ENV.action)
    :param discount
    :param alpha_sched: function mapping timestep to alpha used for exponential weighted update of Q-values
    :param eps: probability of random action
    :param return_time: bool, whether to return convergence time
    :return: Q function, returns for each episode
    """
    nS = env.state.shape[0]
    nA = env.action.shape[0]
    # the difference in norm below which we declare convergence
    eps_convergence = 1e-5

    if Q is None:
        Q = np.zeros((nS, nA))
    rewards = np.zeros((num_episodes, horizon))
    policy = eps_greedy_policy(Q, eps=eps)
    convergence_time = -1

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        t = 0
        Q_prev = Q.copy()
        while not done and t < horizon:
            # take an action
            action_probs = policy(obs)
            action = np.random.choice(env.action, p=action_probs)
            next_obs, reward, done = env.step(action)
            # collect reward
            rewards[ep, t] = reward
            # update Q function
            best_next_action = Q[next_obs].argmax()
            target_q = reward + discount * Q[next_obs][best_next_action]
            alpha = alpha_sched(t)
            Q[obs][action] = (1 - alpha) * Q[obs][action] + alpha * target_q
            # time update
            obs = next_obs
            t += 1
        if np.linalg.norm(Q_prev - Q, ord='fro') < eps_convergence:
            convergence_time = ep

    return (Q, rewards, convergence_time) if return_time else (Q, rewards)


def run_mc_stationary(transition, start, max_steps=np.inf, eps=1e-5):
    prev_state_dist = start.copy()
    t = 0
    while t < max_steps:
        state_dist = transition @ prev_state_dist
        if np.linalg.norm(state_dist - prev_state_dist) < eps:
            converged = True
            print(t, eps)
            break
        prev_state_dist = state_dist
        t += 1
    else:
        converged = False
    return t, converged


def bfs(T, start):
    """
    :param T: 3-dim np.ndarray, deterministic transition matrix indexed by (S, A, S'), basically the graph
    :param start: int, 0-indexed start state
    :return: 1-dim np.ndarray, list of distances to each state, or inf if not reachable from START
    """
    nS, nA = T.shape[:2]

    visited = set()
    visited.add(start)
    queue = Queue(nS)
    queue.put(start)
    distances = np.full(nS, np.inf)
    distances[start] = 0
    while not queue.empty():
        s = queue.get()
        # iterate over states reachable from s
        for nexts in np.nonzero(T[s])[1]:
            if nexts not in visited:
                visited.add(nexts)
                queue.put(nexts)
                distances[nexts] = distances[s] + 1
    return distances
