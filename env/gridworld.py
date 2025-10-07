# file: env/gridworld_fullstate.py
import numpy as np
import random
from typing import Tuple, List, Dict

# ===== Hành động =====
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_TO_VEC = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1)
}

class GridWorld:
    """
    Môi trường GridWorld mô phỏng bài toán MDP mê cung.
    
    Mã ô:
        0 = Tường (không thể đi)
        1 = Đường đi (có thể đi)
        2 = Start
        3 = Goal
        4 = Robot (chỉ hiển thị)
    """

    def __init__(self,
                 grid: np.ndarray,
                 step_reward: float = -1.0,
                 wall_reward: float = -5.0,
                 goal_reward: float = 100.0,
                 max_steps: int = 200,
                 gamma: float = 0.9,
                 seed: int = None):
        self.grid = grid.copy()
        self.n_rows, self.n_cols = grid.shape
        self.step_reward = step_reward
        self.wall_reward = wall_reward
        self.goal_reward = goal_reward
        self.max_steps = max_steps
        self.gamma = gamma
        self.rng = random.Random(seed)

        # Tìm vị trí start/goal
        self.starts = list(zip(*np.where(self.grid == 2)))
        self.goals = list(zip(*np.where(self.grid == 3)))

        if not self.starts:
            raise ValueError("⚠️ Phải có ít nhất một ô Start (2).")
        if not self.goals:
            raise ValueError("⚠️ Phải có ít nhất một ô Goal (3).")

        self.reset()

    # -------------------------
    # Hàm khởi tạo lại môi trường
    # -------------------------
    def reset(self, start_pos: Tuple[int, int] = None) -> np.ndarray:
        if start_pos is None:
            self.pos = self.rng.choice(self.starts)
        else:
            self.pos = tuple(start_pos)
        self.steps = 0
        self.done = False
        return self.get_state()

    # -------------------------
    # Kiểm tra hợp lệ
    # -------------------------
    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n_rows and 0 <= c < self.n_cols

    def is_wall(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r, c] == 0

    def is_goal(self, r: int, c: int) -> bool:
        return self.in_bounds(r, c) and self.grid[r, c] == 3

    # -------------------------
    # Lấy trạng thái full grid
    # -------------------------
    def get_state(self) -> np.ndarray:
        """Trả về toàn bộ ma trận hiện tại với robot đánh dấu 4"""
        grid_state = self.grid.copy()
        r, c = self.pos
        if grid_state[r, c] in [1, 2, 3]:
            grid_state[r, c] = 4
        return grid_state

    # -------------------------
    # Hàm step
    # -------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            return self.get_state(), 0.0, True, {}

        dr, dc = ACTION_TO_VEC[action]
        nr, nc = self.pos[0] + dr, self.pos[1] + dc
        self.steps += 1

        # Nếu đi ra ngoài hoặc đâm vào tường
        if not self.in_bounds(nr, nc) or self.is_wall(nr, nc):
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
        return self.get_state(), reward, done, {}

    # -------------------------
    # Tập trạng thái hợp lệ
    # -------------------------
    def get_all_states(self) -> List[Tuple[int, int]]:
        """Trả về tất cả các ô có thể đi (1, 2, 3)"""
        return [(r, c)
                for r in range(self.n_rows)
                for c in range(self.n_cols)
                if self.grid[r, c] in [1, 2, 3]]

    # -------------------------
    # Hiển thị
    # -------------------------
    def render(self):
        grid_disp = self.get_state()
        print(grid_disp)
