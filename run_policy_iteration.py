from env.gridworld import GridWorld
from algorithms.policy_iteration import policy_iteration   # nếu policy_iteration.py nằm cùng cấp
import numpy as np

grid = np.zeros((5,5))
grid[2,2] = 1
env = GridWorld(grid, start=(0,0), goal=(4,4))

V, policy = policy_iteration(env, gamma=0.9)
print("Giá trị trạng thái:\n", V)
print("Chính sách:\n", policy)