import numpy as np
import pandas as pd
from env.gridworld import GridWorld
from algorithms.q_learning import q_learning
from gui.pygame_gui import run_gui

if __name__ == "__main__":
    grid = np.array([
        [0,0,0,1,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
        [1,1,1,0,1],
        [0,0,0,0,0]
    ])

    env = GridWorld(grid, start=(0,0), goal=(4,4))
    Q = q_learning(env, episodes=500)

    # tìm đường đi tối ưu
    state = env.start
    path = [state]
    done = False
    while not done:
        x,y = state
        action = np.argmax(Q[x, y])
        next_state, _, done = env.step(action)
        state = next_state
        path.append(state)

    # log kết quả
    df = pd.DataFrame(path, columns=["x","y"])
    df.to_csv("logs/results.csv", index=False)

    # hiển thị GUI
    run_gui(env, path)
