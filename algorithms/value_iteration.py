# file: algorithms/value_iteration.py
import numpy as np
from typing import Dict, Tuple

# 4 hành động cơ bản
ACTIONS = [0, 1, 2, 3]  # UP, DOWN, LEFT, RIGHT
ACTION_TO_VEC = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1)
}


def value_iteration(env, gamma=0.9, theta=1e-4, max_iters=10000):
    """
    Value Iteration Algorithm for GridWorld
    env: GridWorld instance
    Returns:
        V: dict[(r,c) -> value]
        policy: dict[(r,c) -> best_action]
    """

    # --- đảm bảo state là tuple hash được ---
    states = [tuple(s) for s in env.get_all_states()]
    V = {s: 0.0 for s in states}

    it = 0
    while True:
        delta = 0.0
        for s in states:
            if env.is_goal(*s):
                continue

            v = V[s]
            max_q = -1e9

            for a in ACTIONS:
                dr, dc = ACTION_TO_VEC[a]
                nr, nc = s[0] + dr, s[1] + dc

                if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                    r = env.wall_reward
                    s_next = s
                else:
                    s_next = (nr, nc)
                    if env.is_goal(nr, nc):
                        r = env.goal_reward
                    else:
                        r = env.step_reward

                # Ép next_state thành tuple để tra dict an toàn
                s_next = tuple(s_next)
                q = r + gamma * V.get(s_next, 0.0)
                max_q = max(max_q, q)

            V[s] = max_q
            delta = max(delta, abs(v - V[s]))

        it += 1
        if delta < theta or it >= max_iters:
            break

    # --- Trích xuất policy ---
    policy = {}
    for s in states:
        if env.is_goal(*s):
            policy[s] = None
            continue

        best_a, best_q = None, -1e9
        for a in ACTIONS:
            dr, dc = ACTION_TO_VEC[a]
            nr, nc = s[0] + dr, s[1] + dc

            if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                r = env.wall_reward
                s_next = s
            else:
                s_next = (nr, nc)
                if env.is_goal(nr, nc):
                    r = env.goal_reward
                else:
                    r = env.step_reward

            s_next = tuple(s_next)
            q = r + gamma * V.get(s_next, 0.0)
            if q > best_q:
                best_q, best_a = q, a

        policy[s] = best_a

    return V, policy
