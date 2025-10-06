# file: utils/visualize.py
import matplotlib.pyplot as plt
import numpy as np
import os

ACTION_TO_VEC = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1)}
ACTION_ARROW = {0: (0, 0.4), 1: (0, -0.4), 2: (-0.4, 0), 3: (0.4, 0)}
ACTION_TO_CHAR = {0:'↑',1:'↓',2:'←',3:'→'}

def plot_grid(grid, agent_pos=None, title="GridWorld", savepath=None):
    rows, cols = grid.shape
    plt.figure(figsize=(cols/2, rows/2))
    cmap = plt.get_cmap('tab20')
    # draw squares
    for r in range(rows):
        for c in range(cols):
            val = grid[r,c]
            if val == 1:
                color = 'black'
            elif val == 0:
                color = 'white'
            elif val == 2:
                color = 'green'
            elif val == 3:
                color = 'red'
            plt.gca().add_patch(plt.Rectangle((c, rows-1-r), 1, 1, edgecolor='gray', facecolor=color))
    # agent
    if agent_pos is not None:
        ar, ac = agent_pos
        plt.text(ac+0.5, rows-1-ar+0.5, 'R', va='center', ha='center', fontsize=14, color='blue')
    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().set_aspect('equal')
    plt.gca().axis('off')
    plt.title(title)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def draw_policy(grid, policy, start, goal, title="Optimal Path", savepath=None):
    rows, cols = grid.shape
    plt.figure(figsize=(cols/2, rows/2))

    # Vẽ ô
    for r in range(rows):
        for c in range(cols):
            val = grid[r, c]
            if val == 1:
                color = 'black'  # tường
            elif val == 2:
                color = 'green'  # goal
            elif val == 3:
                color = 'red'    # start (tùy bạn đặt)
            else:
                color = 'white'
            plt.gca().add_patch(plt.Rectangle((c, rows-1-r), 1, 1, edgecolor='gray', facecolor=color))

    # Truy vết đường đi theo policy
    path = []
    s = start
    visited = set()
    while s != goal and s not in visited:
        visited.add(s)
        path.append(s)
        a = policy.get(s, None)
        if a is None:
            break
        dr, dc = ACTION_TO_VEC[a]
        s = (s[0] + dr, s[1] + dc)
        if not (0 <= s[0] < rows and 0 <= s[1] < cols):
            break
    path.append(goal)

    # Vẽ đường đi
    if len(path) > 1:
        xs = [c + 0.5 for (r, c) in path]
        ys = [rows - 1 - r + 0.5 for (r, c) in path]
        plt.plot(xs, ys, 'b-', linewidth=2.5, label='Optimal Path')
        plt.scatter(xs[0], ys[0], color='red', s=100, label='Start')
        plt.scatter(xs[-1], ys[-1], color='green', s=100, label='Goal')

    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().set_aspect('equal')
    plt.gca().axis('off')
    plt.title(title)
    plt.legend()

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()

def plot_learning_curve(rewards, title="Learning Curve", savepath=None):
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.arange(len(rewards))
    plt.figure(figsize=(8,4))
    plt.plot(x, rewards, label='episode reward')
    # moving average
    if len(rewards) >= 50:
        ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(len(ma))+49, ma, label='MA(50)')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(title)
    plt.legend()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
