# file: utils/visualize.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os


# ======================
# 🔹 Hàm tiện ích chung
# ======================
def _init_figure(grid_shape, figsize_ratio=0.5):
    """Tạo figure và axes có tỉ lệ phù hợp với kích thước bản đồ."""
    rows, cols = grid_shape
    fig_width = max(8, cols * figsize_ratio)
    fig_height = max(4, rows * figsize_ratio)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    return fig, ax


def _save_plot(savepath):
    """Lưu hình ảnh nếu có đường dẫn."""
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
        print(f"✅ Đã lưu hình vào: {savepath}")


# ======================
# 🔹 1. Vẽ bản đồ Grid
# ======================
def plot_grid(grid, title="", ax=None, savepath=None):
    """
    Vẽ bản đồ lưới (grid world) với màu sắc quy ước:
      0: trắng (tường)
      1: đen (đường đi)
      2: xanh lá (start)
      3: đỏ (goal)
    """
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)

    if ax is None:
        _, ax = _init_figure(grid.shape)

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

    _save_plot(savepath)
    return ax


# ======================
# 🔹 2. Vẽ chính sách Policy
# ======================
def draw_policy(grid, policy, title="Policy", savepath=None):
    """
    Vẽ toàn bộ chính sách (policy) lên bản đồ:
    - Mỗi ô đường đi (1) hiển thị mũi tên hành động.
    """
    if not policy:
        print("⚠️ Không có policy để hiển thị.")
        return

    fig, ax = _init_figure(grid.shape)
    plot_grid(grid, title=title, ax=ax)

    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for (r, c), action in policy.items():
        if grid[r, c] == 1:
            ax.text(c, r, arrow_map.get(action, '?'),
                    ha='center', va='center',
                    color='white', fontsize=12, fontweight='bold')

    _save_plot(savepath)
    plt.show()


# ======================
# 🔹 3. Vẽ đường đi tối ưu
# ======================
def draw_optimal_path(grid, policy, start_pos, title="Optimal Path", savepath=None):
    """
    Vẽ đường đi tối ưu từ điểm bắt đầu (start) đến đích (goal)
    theo policy tối ưu đã học.
    """
    if not policy:
        print(f"⚠️ Không thể vẽ '{title}' vì không có policy.")
        return

    fig, ax = _init_figure(grid.shape)
    plot_grid(grid, title=title, ax=ax)

    rows, cols = grid.shape
    current = start_pos
    path = [current]
    action_to_vec = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    # Tạo đường đi dựa vào policy
    for _ in range(rows * cols):
        action = policy.get(current)
        if action is None or grid[current] == 3:
            break
        dr, dc = action_to_vec[action]
        next_pos = (current[0] + dr, current[1] + dc)
        path.append(next_pos)
        current = next_pos

    arrow_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for (r, c) in path[:-1]:
        if grid[r, c] != 2:  # bỏ qua ô start
            action = policy.get((r, c))
            if action is not None:
                ax.text(c, r, arrow_map[action],
                        ha='center', va='center',
                        color='cyan', fontsize=12, fontweight='bold')

    _save_plot(savepath)
    plt.show()


# ======================
# 🔹 4. Vẽ biểu đồ học tập
# ======================
def plot_learning_curve(rewards, title="Learning Curve", savepath=None):
    """
    Vẽ biểu đồ học (learning curve) thể hiện tổng phần thưởng qua các tập.
    """
    if len(rewards) == 0:
        print("⚠️ Không có dữ liệu phần thưởng để vẽ.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(rewards, alpha=0.6, label='Phần thưởng mỗi tập')

    # Trung bình trượt để làm mượt biểu đồ
    if len(rewards) >= 50:
        ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
        plt.plot(np.arange(len(ma)) + 49, ma, color='red', linewidth=2,
                 label='Trung bình trượt (50 tập)')

    plt.title(title, fontsize=16)
    plt.xlabel("Tập (Episode)", fontsize=12)
    plt.ylabel("Tổng phần thưởng (Total Reward)", fontsize=12)
    plt.grid(True)
    plt.legend()

    _save_plot(savepath)
    plt.show()
