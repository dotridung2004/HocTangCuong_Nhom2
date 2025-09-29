# file: env/gridworld.py
import numpy as np
import json
import os
import random
from typing import Tuple, List

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_TO_VEC = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}

class GridWorld:
    """
    Deterministic GridWorld environment.
    Map encoding:
      0 = free
      1 = wall
      2 = start
      3 = goal
    Robot position tracked separately (r, c).
    """
    def __init__(self, grid: np.ndarray, step_reward=-1, wall_reward=-5, goal_reward=100, max_steps=500, seed=None):
        self.grid = grid.copy()
        self.n_rows, self.n_cols = grid.shape
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.seed = seed
        self.rng = random.Random(seed)
        # find start(s) and goal(s)
        self.starts = list(zip(*np.where(self.grid == 2)))
        self.goals = list(zip(*np.where(self.grid == 3)))
        if not self.starts:
            raise ValueError("Map must contain at least one start cell (value 2).")
        if not self.goals:
            raise ValueError("Map must contain at least one goal cell (value 3).")
        self.reset()

    @classmethod
    def from_json(cls, path, **kwargs):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        grid = np.array(data['grid'], dtype=int)
        return cls(grid, **kwargs)

    def reset(self, start_pos=None):
        # place robot at start (either specified or random among starts)
        if start_pos is None:
            self.pos = self.rng.choice(self.starts)
        else:
            self.pos = tuple(start_pos)
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        # state as (r,c)
        return self.pos

    def in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_wall(self, r, c):
        if not self.in_bounds(r, c):
            return True
        return self.grid[r, c] == 1

    def is_goal(self, r, c):
        return self.in_bounds(r, c) and self.grid[r, c] == 3

    def step(self, action: int) -> Tuple[Tuple[int,int], float, bool, dict]:
        """
        Perform action (deterministic). Return (next_state, reward, done, info)
        """
        if self.done:
            return self._get_state(), 0.0, True, {}

        dr, dc = ACTION_TO_VEC[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1

        # hitting wall or out of bounds
        if not self.in_bounds(nr, nc) or self.grid[nr, nc] == 1:
            reward = self.wall_reward
            next_pos = self.pos  # no movement
            done = False
        else:
            next_pos = (nr, nc)
            if self.is_goal(nr, nc):
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False

        # step count limit
        if self.steps >= self.max_steps:
            done = True

        self.pos = next_pos
        self.done = done
        return self._get_state(), float(reward), done, {}

    def get_all_states(self):
        """Return list of all free/start/goal coordinate tuples (non-wall)."""
        coords = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r, c] != 1:
                    coords.append((r, c))
        return coords

    def save_to_json(self, path):
        data = {'grid': self.grid.tolist()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
