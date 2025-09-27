import numpy as np

class GridWorld:
    def __init__(self, grid, start=(0,0), goal=(4,4)):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.n_actions = 4  # up, down, left, right
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:    # up
            x = max(x-1, 0)
        elif action == 1:  # down
            x = min(x+1, self.grid.shape[0]-1)
        elif action == 2:  # left
            y = max(y-1, 0)
        elif action == 3:  # right
            y = min(y+1, self.grid.shape[1]-1)

        # kiểm tra tường
        if self.grid[x, y] == 1:
            x, y = self.state

        self.state = (x, y)

        # tính reward
        if self.state == self.goal:
            return self.state, 1, True
        else:
            return self.state, -0.01, False
