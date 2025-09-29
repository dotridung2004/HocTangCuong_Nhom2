# file: algorithms/value_iteration.py
import numpy as np
from typing import Tuple, Dict

ACTIONS = [0,1,2,3]  # UP, DOWN, LEFT, RIGHT
ACTION_TO_VEC = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}

def value_iteration(env, gamma=0.9, theta=1e-4, max_iters=10000):
    """
    env: GridWorld instance
    Return V dict mapping (r,c) -> value and policy mapping (r,c) -> best action
    """
    states = env.get_all_states()
    V = {s: 0.0 for s in states}
    it = 0
    while True:
        delta = 0.0
        for s in states:
            # skip terminal (goal)
            if env.is_goal(*s):
                continue
            v = V[s]
            max_q = -1e9
            for a in ACTIONS:
                # deterministic next
                dr,dc = ACTION_TO_VEC[a]
                nr, nc = s[0]+dr, s[1]+dc
                if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                    r = env.wall_reward
                    s_next = s
                    done = False
                else:
                    s_next = (nr,nc)
                    if env.is_goal(nr,nc):
                        r = env.goal_reward
                    else:
                        r = env.step_reward
                q = r + gamma * V.get(s_next,0.0)
                if q > max_q:
                    max_q = q
            V[s] = max_q
            delta = max(delta, abs(v - V[s]))
        it += 1
        if delta < theta or it >= max_iters:
            break

    # extract policy
    policy = {}
    for s in states:
        if env.is_goal(*s):
            policy[s] = None
            continue
        best_a = None
        best_q = -1e9
        for a in ACTIONS:
            dr,dc = ACTION_TO_VEC[a]
            nr, nc = s[0]+dr, s[1]+dc
            if not env.in_bounds(nr, nc) or env.is_wall(nr, nc):
                r = env.wall_reward
                s_next = s
            else:
                s_next = (nr,nc)
                if env.is_goal(nr,nc):
                    r = env.goal_reward
                else:
                    r = env.step_reward
            q = r + gamma * V.get(s_next, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        policy[s] = best_a
    return V, policy
