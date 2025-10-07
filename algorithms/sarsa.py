# file: algorithms/sarsa.py
import numpy as np
import os
import pickle
from typing import Tuple

ACTIONS = [0, 1, 2, 3]
ACTION_TO_VEC = {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1)}

# -------------------------
# Chuyển trạng thái grid thành key hashable
# -------------------------
def state_to_key(state: np.ndarray) -> Tuple[Tuple[int,...], ...]:
    return tuple(map(tuple, state))

# -------------------------
# Khởi tạo Q-table
# -------------------------
def init_q(env):
    states = env.get_all_states()
    Q = {}
    for s in states:
        Q[s] = np.zeros(len(ACTIONS), dtype=float)
    return Q

# -------------------------
# Epsilon-greedy
# -------------------------
def epsilon_greedy(Q, state, epsilon, rng):
    state_key = state_to_key(state)
    if state_key not in Q:
        Q[state_key] = np.zeros(len(ACTIONS), dtype=float)
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    else:
        qvals = Q[state_key]
        return int(np.argmax(qvals))

# -------------------------
# SARSA chính
# -------------------------
def sarsa(env,
          num_episodes=2000,
          alpha=0.1,
          gamma=0.9,
          epsilon=1.0,
          epsilon_decay=0.999,
          min_epsilon=0.05,
          seed=None,
          save_path=None):
    
    rng = np.random.RandomState(seed)
    Q = {}  # Q-table trống
    rewards_per_episode = []
    steps_per_episode = []

    for ep in range(num_episodes):
        state = env.reset()
        state_key = state_to_key(state)
        if state_key not in Q:
            Q[state_key] = np.zeros(len(ACTIONS), dtype=float)
        
        a = epsilon_greedy(Q, state, epsilon, rng)

        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            next_state, r, done, _ = env.step(a)
            next_state_key = state_to_key(next_state)
            if next_state_key not in Q:
                Q[next_state_key] = np.zeros(len(ACTIONS), dtype=float)

            next_a = epsilon_greedy(Q, next_state, epsilon, rng) if not done else None

            # SARSA update
            if not done:
                Q[state_key][a] += alpha * (r + gamma * Q[next_state_key][next_a] - Q[state_key][a])
            else:
                Q[state_key][a] += alpha * (r - Q[state_key][a])

            state = next_state
            state_key = next_state_key
            a = next_a

            total_reward += r
            steps += 1

        rewards_per_episode.append(total_reward)
        steps_per_episode.append(steps)

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Lưu Q-table nếu cần
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path,'wb') as f:
            pickle.dump(Q, f)

    return Q, rewards_per_episode, steps_per_episode
