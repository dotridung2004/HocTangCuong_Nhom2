# file: algorithms/q_learning.py
import numpy as np
import os
import pickle
from typing import Tuple
ACTIONS = [0,1,2,3]
ACTION_TO_VEC = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}

def init_q(env):
    states = env.get_all_states()
    Q = {}
    for s in states:
        Q[s] = np.zeros(len(ACTIONS), dtype=float)
    return Q

def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    else:
        qvals = Q[state]
        return int(np.argmax(qvals))

def q_learning(env, num_episodes=2000, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.999, min_epsilon=0.05, seed=None, save_path=None):
    rng = np.random.RandomState(seed)
    Q = init_q(env)
    rewards_per_episode = []
    steps_per_episode = []
    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        while not done:
            a = epsilon_greedy(Q, state, epsilon, rng)
            next_state, r, done, _ = env.step(a)
            # Q update
            best_next = np.max(Q[next_state])
            Q[state][a] += alpha * (r + gamma * best_next - Q[state][a])
            state = next_state
            total_reward += r
            steps += 1
        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)
        # decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path,'wb') as f:
            pickle.dump(Q, f)
    return Q, rewards_per_episode, steps_per_episode
