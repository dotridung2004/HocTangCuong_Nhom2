# file: main.py
import os
import numpy as np
from env.gridworld import GridWorld
from utils.map_io import load_map_json, save_map_json
from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from utils.visualize import plot_grid, draw_policy, plot_learning_curve


def choose_start_goal(grid, save_path="maps/dhtl_map.json"):
    """Cho người dùng nhập điểm Start/Goal rồi cập nhật bản đồ."""
    rows, cols = grid.shape
    print(f"Bản đồ có kích thước: {rows} x {cols}")
    print("⚠ Lưu ý: chỉ số bắt đầu từ 0 (vd: ô trên cùng bên trái là (0,0))")

    try:
        sr = int(input("Nhập hàng (row) điểm START: "))
        sc = int(input("Nhập cột (col) điểm START: "))
        gr = int(input("Nhập hàng (row) điểm GOAL: "))
        gc = int(input("Nhập cột (col) điểm GOAL: "))
    except ValueError:
        print("❌ Lỗi: bạn phải nhập số nguyên.")
        return grid

    # Xóa các ô start/goal cũ
    grid[grid == 2] = 0
    grid[grid == 3] = 0

    # Gán start/goal mới
    if 0 <= sr < rows and 0 <= sc < cols:
        grid[sr, sc] = 2
    if 0 <= gr < rows and 0 <= gc < cols:
        grid[gr, gc] = 3

    # Lưu lại map
    save_map_json(grid, save_path)
    print(f"✅ Đã cập nhật Start={sr,sc}, Goal={gr,gc} trong {save_path}")
    return grid


def run_dp(env):
    V_vi, policy_vi = value_iteration(env, gamma=0.9)
    draw_policy(env.grid, policy_vi, title="Policy - Value Iteration", savepath="results/policy_vi.png")

    V_pi, policy_pi = policy_iteration(env, gamma=0.9)
    draw_policy(env.grid, policy_pi, title="Policy - Policy Iteration", savepath="results/policy_pi.png")

    return policy_vi, policy_pi


def run_q_learning(env):
    Q, rewards, steps = q_learning(env, num_episodes=1500, alpha=0.1, gamma=0.9,
                                   epsilon=1.0, epsilon_decay=0.995, seed=42)
    policy = {s: int(np.argmax(arr)) for s, arr in Q.items()}
    draw_policy(env.grid, policy, title="Policy - Q-learning", savepath="results/policy_qlearning.png")
    plot_learning_curve(rewards, title="Q-learning rewards", savepath="results/learning_qlearning.png")
    return policy


def run_sarsa(env):
    Q, rewards, steps = sarsa(env, num_episodes=1500, alpha=0.1, gamma=0.9,
                              epsilon=1.0, epsilon_decay=0.995, seed=24)
    policy = {s: int(np.argmax(arr)) for s, arr in Q.items()}
    draw_policy(env.grid, policy, title="Policy - SARSA", savepath="results/policy_sarsa.png")
    plot_learning_curve(rewards, title="SARSA rewards", savepath="results/learning_sarsa.png")
    return policy


def evaluate_policy(env, policy, episodes=50):
    successes, steps_list = 0, []
    for ep in range(episodes):
        state = env.reset()
        done, steps = False, 0
        while not done:
            a = policy.get(state, None)
            if a is None: break
            state, r, done, _ = env.step(a)
            steps += 1
            if steps > env.max_steps: break
        if env.is_goal(*state):
            successes += 1
            steps_list.append(steps)
    return successes / episodes * 100, np.mean(steps_list) if steps_list else None


def main():
    # --- Load map ---
    grid = load_map_json("maps/dhtl_map.json")

    # --- Người dùng nhập Start/Goal ---
    grid = choose_start_goal(grid, save_path="maps/dhtl_map.json")

    # --- Khởi tạo môi trường ---
    env = GridWorld(grid, step_reward=-1, wall_reward=-5, goal_reward=100, max_steps=800, seed=123)
    plot_grid(env.grid, title="DHTL Map (User-defined Start/Goal)", savepath="results/map.png")

    # --- Thuật toán ---
    print("\n▶ Value Iteration & Policy Iteration...")
    pi_vi, pi_pi = run_dp(env)

    print("\n▶ Q-learning...")
    pi_ql = run_q_learning(env)

    print("\n▶ SARSA...")
    pi_sarsa = run_sarsa(env)

    # --- Đánh giá ---
    print("\n📊 Kết quả đánh giá:")
    for name, policy in [("Value Iteration", pi_vi),
                         ("Policy Iteration", pi_pi),
                         ("Q-learning", pi_ql),
                         ("SARSA", pi_sarsa)]:
        sr, avg = evaluate_policy(env, policy, episodes=50)
        print(f"- {name:16}: Success {sr:.1f}% | Avg steps {avg}")


if __name__ == "__main__":
    main()
