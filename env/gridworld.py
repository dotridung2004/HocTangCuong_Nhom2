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
    Môi trường GridWorld.
    # === SỬA LẠI: Cập nhật quy ước map cho đúng với dự án ===
    Quy ước map:
      0 = tường (vật cản)
      1 = đường đi
      2 = điểm bắt đầu
      3 = điểm kết thúc
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
        
        self.starts = list(zip(*np.where(self.grid == 2)))
        self.goals = list(zip(*np.where(self.grid == 3)))
        
        if not self.starts:
            raise ValueError("Bản đồ phải chứa ít nhất một ô bắt đầu (giá trị 2).")
        if not self.goals:
            raise ValueError("Bản đồ phải chứa ít nhất một ô kết thúc (giá trị 3).")
            
        self.reset()

    def reset(self, start_pos=None):
        if start_pos is None:
            self.pos = self.rng.choice(self.starts)
        else:
            self.pos = tuple(start_pos)
        self.steps = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return self.pos

    def in_bounds(self, r, c):
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_wall(self, r, c):
        if not self.in_bounds(r, c):
            return True
        # === SỬA LẠI: Tường là ô có giá trị 0 ===
        return self.grid[r, c] == 0

    def is_goal(self, r, c):
        return self.in_bounds(r, c) and self.grid[r, c] == 3

    def step(self, action: int) -> Tuple[Tuple[int,int], float, bool, dict]:
        if self.done:
            return self._get_state(), 0.0, True, {}

        dr, dc = ACTION_TO_VEC[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1

        # === SỬA LẠI: Kiểm tra va chạm với tường (giá trị 0) ===
        if not self.in_bounds(nr, nc) or self.grid[nr, nc] == 0:
            reward = self.wall_reward
            next_pos = self.pos
            done = False
        else:
            next_pos = (nr, nc)
            if self.is_goal(nr, nc):
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

    def get_all_states(self):
        """Trả về danh sách tất cả các ô không phải là tường."""
        coords = []
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                # === SỬA LẠI: Trạng thái hợp lệ là các ô không phải tường (giá trị 0) ===
                if self.grid[r, c] != 0:
                    coords.append((r, c))
        return coords