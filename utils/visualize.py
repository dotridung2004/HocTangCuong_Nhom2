# file: utils/visualize.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def plot_grid(grid, title="", ax=None, savepath=None):
    """
    Vẽ bản đồ grid một cách hiệu quả với màu sắc tùy chỉnh.
    """
    if ax is None:
        rows, cols = grid.shape
        figsize_ratio = 0.5
        fig_width, fig_height = max(8, cols * figsize_ratio), max(4, rows * figsize_ratio)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Quy ước màu: 0:trắng (tường), 1:đen (đường đi), 2:xanh lá (start), 3:đỏ (goal)
    cmap = ListedColormap(['white', 'black', 'green', 'red'])
    
    ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=3)

    rows, cols = grid.shape
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16)

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    
    return ax

def draw_policy(grid, policy, title="Policy", savepath=None):
    """
    Hàm vẽ toàn bộ policy (mũi tên ở tất cả các ô đường đi).
    """
    rows, cols = grid.shape
    figsize_ratio = 0.5
    fig_width, fig_height = max(8, cols * figsize_ratio), max(4, rows * figsize_ratio)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_grid(grid, title=title, ax=ax)
    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}

    for state, action in policy.items():
        r, c = state
        if grid[r, c] == 1: # Giả định đường đi là 1
            ax.text(c, r, arrow_map.get(action, '?'), ha='center', va='center',
                    color='white', fontsize=10, fontweight='bold')

    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        print(f"✅ Đã lưu policy vào: {savepath}")
    plt.show()

def draw_optimal_path(grid, policy, start_pos, title="Optimal Path", savepath=None):
    """
    Vẽ con đường tối ưu từ start đến goal với các mũi tên chỉ hướng.
    """
    if not policy:
        print(f"⚠️ Không thể vẽ đường đi cho '{title}' vì không có policy.")
        return

    rows, cols = grid.shape
    figsize_ratio = 0.5
    fig_width, fig_height = max(8, cols * figsize_ratio), max(4, rows * figsize_ratio)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    plot_grid(grid, title=title, ax=ax)
    
    path = []
    current_pos = start_pos
    path.append(current_pos)
    
    for _ in range(rows * cols): 
        action = policy.get(current_pos, None)
        if action is None or grid[current_pos] == 3:
            break
        
        dr, dc = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[action]
        next_pos = (current_pos[0] + dr, current_pos[1] + dc)
        path.append(next_pos)
        current_pos = next_pos

    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for i in range(len(path) - 1):
        r, c = path[i]
        action = policy.get((r, c), None)
        
        if action is not None and grid[r, c] != 2:
            arrow = arrow_map.get(action)
            ax.text(c, r, arrow, ha='center', va='center',
                    color='cyan', fontsize=12, fontweight='bold')
            
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        print(f"✅ Đã lưu đường đi tối ưu vào: {savepath}")
        
    plt.show()

def plot_learning_curve(rewards, title="Learning Curve", savepath=None):
    """
    Vẽ biểu đồ học (learning curve).
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.6, label='Phần thưởng mỗi tập')
    if len(rewards) >= 50:
        ma_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(len(ma_rewards)) + 49, ma_rewards, color='red', linewidth=2, label='Trung bình trượt (50 tập)')
    plt.title(title, fontsize=16)
    plt.xlabel("Tập (Episode)", fontsize=12)
    plt.ylabel("Tổng phần thưởng (Total Reward)", fontsize=12)
    plt.grid(True)
    plt.legend()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
        print(f"✅ Đã lưu learning curve vào: {savepath}")
    plt.show()