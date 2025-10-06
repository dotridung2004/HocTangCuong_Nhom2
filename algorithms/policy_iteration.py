# file: algorithms/policy_iteration.py
import numpy as np
from typing import Dict

ACTIONS = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
ACTION_TO_VEC = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}


def policy_evaluation(env, policy: Dict, gamma=0.9, theta=1e-4, max_iters=10000):
    """Đánh giá giá trị trạng thái V dưới policy hiện tại"""
    states = [tuple(map(int, s)) for s in env.get_all_states()]  # đảm bảo tuple key
    V = {s: 0.0 for s in states}
    it = 0

    while True:
        delta = 0.0
        for s in states:
            if env.is_goal(*s):
                continue
            v = V[s]
            a = policy[s]
            dr, dc = ACTION_TO_VEC[a]
            nr, nc = s[0] + dr, s[1] + dc

            if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                r = env.wall_reward
                s_next = s
            else:
                s_next = (nr, nc)
                r = env.goal_reward if env.is_goal(nr, nc) else env.step_reward

            V[s] = r + gamma * V.get(s_next, 0.0)
            delta = max(delta, abs(v - V[s]))

        it += 1
        if delta < theta or it >= max_iters:
            break
    return V


def policy_iteration(env, gamma=0.9, theta=1e-4, max_iters=100):
    """Thuật toán Policy Iteration"""
    states = [tuple(map(int, s)) for s in env.get_all_states()]  # convert all to tuple
    policy = {}

    # Khởi tạo policy ngẫu nhiên
    for s in states:
        if env.is_goal(*s):
            policy[s] = None
        else:
            policy[s] = np.random.choice(ACTIONS)

    it = 0
    while True:
        # Bước 1: Policy Evaluation
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)

        # Bước 2: Policy Improvement
        policy_stable = True
        for s in states:
            if env.is_goal(*s):
                continue

            old_action = policy[s]
            best_a = None
            best_q = -1e9

            for a in ACTIONS:
                dr, dc = ACTION_TO_VEC[a]
                nr, nc = s[0] + dr, s[1] + dc

                if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                    r = env.wall_reward
                    s_next = s
                else:
                    s_next = (nr, nc)
                    r = env.goal_reward if env.is_goal(nr, nc) else env.step_reward

                q = r + gamma * V.get(s_next, 0.0)
                if q > best_q:
                    best_q = q
                    best_a = a

            policy[s] = best_a
            if old_action != best_a:
                policy_stable = False

        it += 1
        if policy_stable or it >= max_iters:
            break

    return V, policy
