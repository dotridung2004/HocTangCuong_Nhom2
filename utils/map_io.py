# file: utils/map_io.py
import numpy as np
import json
import os

def sample_map(rows=10, cols=14):
    # create simple map with walls around and some internal walls
    grid = np.zeros((rows, cols), dtype=int)
    # outer walls
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    # add some random walls (deterministic for sample)
    walls = [(3,3),(3,4),(3,5),(5,8),(6,8),(4,10),(2,10)]
    for (r,c) in walls:
        if 0<=r<rows and 0<=c<cols:
            grid[r,c] = 1
    # start and goal
    grid[1,1] = 2
    grid[rows-2, cols-2] = 3
    return grid

def save_map_json(grid, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'grid': grid.tolist()}, f, indent=2, ensure_ascii=False)

def load_map_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    import numpy as np
    return np.array(data['grid'], dtype=int)
