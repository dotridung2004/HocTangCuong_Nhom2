# ===============================
# file: main.py
# ===============================
import os
import numpy as np
from env.gridworld import GridWorld
from utils.map_io import load_map_json
from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from utils.visualize import plot_grid, draw_policy, plot_learning_curve


# ===============================
# 1️⃣ Chạy các thuật toán Dynamic Programming
# ===============================
def run_dp(env):
    print("▶ Running Value Iteration...")
    V_vi, policy_vi = value_iteration(env, gamma=0.9)
    draw_policy(env.grid, policy_vi,
                title="Policy - Value Iteration",
                savepath="results/policy_vi.png")

    print("▶ Running Policy Iteration...")
    V_pi, policy_pi = policy_iteration(env, gamma=0.9)
    draw_policy(env.grid, policy_pi,
                title="Policy - Policy Iteration",
                savepath="results/policy_pi.png")

    return policy_vi, policy_pi


# ===============================
# 2️⃣ Q-learning
# ===============================
def run_q_learning(env):
    print("▶ Running Q-learning...")
    Q, rewards, steps = q_learning(
        env,
        num_episodes=1500,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        seed=42
    )
    policy = {s: int(np.argmax(arr)) for s, arr in Q.items()}
    draw_policy(env.grid, policy,
                title="Policy - Q-learning",
                savepath="results/policy_qlearning.png")
    plot_learning_curve(rewards,
                        title="Q-learning Rewards",
                        savepath="results/learning_qlearning.png")
    return policy


# ===============================
# 3️⃣ SARSA
# ===============================
def run_sarsa(env):
    print("▶ Running SARSA...")
    Q, rewards, steps = sarsa(
        env,
        num_episodes=1500,
        alpha=0.1,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        seed=24
    )
    policy = {s: int(np.argmax(arr)) for s, arr in Q.items()}
    draw_policy(env.grid, policy,
                title="Policy - SARSA",
                savepath="results/policy_sarsa.png")
    plot_learning_curve(rewards,
                        title="SARSA Rewards",
                        savepath="results/learning_sarsa.png")
    return policy


# ===============================
# 4️⃣ Đánh giá chính sách
# ===============================
def evaluate_policy(env, policy, episodes=50):
    successes, steps_list = 0, []
    for ep in range(episodes):
        state = env.reset()
        done, steps = False, 0
        while not done:
            a = policy.get(state, None)
            if a is None:
                break
            state, r, done, _ = env.step(a)
            steps += 1
            if steps > env.max_steps:
                break
        # nếu chạm đích
        if tuple(env.pos) in env.goals:
            successes += 1
            steps_list.append(steps)
    success_rate = successes / episodes * 100
    avg_steps = np.mean(steps_list) if steps_list else None
    return success_rate, avg_steps


# ===============================
# 5️⃣ Chương trình chính
# ===============================
def main():
    os.makedirs("results", exist_ok=True)

    map_path = "maps/dhtl_map.json"
    if not os.path.exists(map_path):
        print(f"❌ Không tìm thấy file bản đồ: {map_path}")
        print("👉 Hãy chạy convert_map.py trước để tạo file JSON từ ảnh.")
        return

    # --- Load map ---
    print(f"📂 Loading map from: {map_path}")
    grid = load_map_json(map_path)

    # --- Khởi tạo môi trường ---
    env = GridWorld(
        grid,
        step_reward=-1,
        wall_reward=-5,
        goal_reward=100,
        max_steps=800,
        seed=123
    )

    # --- Vẽ bản đồ ---
    plot_grid(env.base_grid, title="DHTL Map (Start & Goal)", savepath="results/map.png")

    # --- Chạy Value Iteration + Policy Iteration ---
    pi_vi, pi_pi = run_dp(env)

    # --- Chạy Q-learning & SARSA ---
    pi_ql = run_q_learning(env)
    pi_sarsa = run_sarsa(env)

    # --- Đánh giá ---
    print("\n📊 Đánh giá các chính sách:")
    for name, policy in [
        ("Value Iteration", pi_vi),
        ("Policy Iteration", pi_pi),
        ("Q-learning", pi_ql),
        ("SARSA", pi_sarsa)
    ]:
        sr, avg = evaluate_policy(env, policy, episodes=50)
        print(f"- {name:16}: Success {sr:.1f}% | Avg steps {avg}")


# ===============================
# 6️⃣ Entry point
# ===============================
if __name__ == "__main__":
    main()
