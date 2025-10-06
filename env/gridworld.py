# env/gridworld.py
import numpy as np
import json
import os
import random
from typing import Tuple, List, Dict

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
    Deterministic GridWorld environment:
      0 = Free
      1 = Wall
      2 = Start
      3 = Goal
      4 = Robot
    """
    def __init__(self, grid: np.ndarray, step_reward=-1, wall_reward=-5, goal_reward=100, 
                 max_steps=500, gamma=0.9, seed=None):
        self.base_grid = grid.copy()
        self.n_rows, self.n_cols = grid.shape
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.gamma = gamma
        self.seed = seed
        self.rng = random.Random(seed)

        # tìm start và goal
        self.starts = list(zip(*np.where(self.base_grid == 2)))
        self.goals = list(zip(*np.where(self.base_grid == 3)))
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
        if start_pos is None:
            self.pos = self.rng.choice(self.starts)
        else:
            self.pos = tuple(start_pos)
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        grid_with_robot = self.base_grid.copy()
        r, c = self.pos
        grid_with_robot[r, c] = 4
        return grid_with_robot

    def in_bounds(self, r, c) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}

        dr, dc = ACTION_TO_VEC[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1

        if not self.in_bounds(nr, nc) or self.base_grid[nr, nc] == 1:
            reward = self.wall_reward
            next_pos = self.pos
            done = False
        else:
            next_pos = (nr, nc)
            if self.base_grid[nr, nc] == 3:  # Goal
                reward = self.goal_reward
                done = True
            else:
                reward = self.step_reward
                done = False

        if self.steps >= self.max_steps:
            done = True

        self.pos = next_pos
        self.done = done
        return self._get_state(), float(reward), done, {}

    def get_all_states(self) -> List[np.ndarray]:
        states = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.base_grid[r, c] != 1:  # không phải wall
                    g = self.base_grid.copy()
                    g[r, c] = 4
                    states.append(g)
        return states

    def save_to_json(self, path):
        data = {'grid': self.base_grid.tolist()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
