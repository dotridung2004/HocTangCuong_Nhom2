# file: main.py
import os
import numpy as np
from env.gridworld import GridWorld
from utils.map_io import load_map_json, sample_map, save_map_json
from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning
from algorithms.sarsa import sarsa
from utils.visualize import plot_grid, draw_policy, plot_learning_curve

def ensure_map():
    if not os.path.exists("maps/campus_map.json"):
        grid = sample_map(12,18)
        save_map_json(grid, "maps/campus_map.json")
        print("Saved sample map to maps/campus_map.json")

def run_dp(env):
    V_vi, policy_vi = value_iteration(env, gamma=0.9)
    print("Value Iteration done.")
    draw_policy(env.grid, policy_vi, title="Policy - Value Iteration", savepath="results/policy_vi.png")
    V_pi, policy_pi = policy_iteration(env, gamma=0.9)
    print("Policy Iteration done.")
    draw_policy(env.grid, policy_pi, title="Policy - Policy Iteration", savepath="results/policy_pi.png")

def run_q_learning(env):
    Q, rewards, steps = q_learning(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, seed=42, save_path="results/qtable_qlearning.pkl")
    # derive policy
    policy = {}
    for s, arr in Q.items():
        policy[s] = int(np.argmax(arr))
    draw_policy(env.grid, policy, title="Policy - Q-learning", savepath="results/policy_qlearning.png")
    plot_learning_curve(rewards, title="Q-learning rewards", savepath="results/learning_qlearning.png")
    print("Q-learning done.")

def run_sarsa(env):
    Q, rewards, steps = sarsa(env, num_episodes=1500, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.995, seed=24, save_path="results/qtable_sarsa.pkl")
    policy = {}
    for s, arr in Q.items():
        policy[s] = int(np.argmax(arr))
    draw_policy(env.grid, policy, title="Policy - SARSA", savepath="results/policy_sarsa.png")
    plot_learning_curve(rewards, title="SARSA rewards", savepath="results/learning_sarsa.png")
    print("SARSA done.")

def evaluate_policy(env, policy, episodes=50):
    successes = 0
    steps_list = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            a = policy.get(state, None)
            if a is None:
                break
            state, r, done, _ = env.step(a)
            steps += 1
            if steps > env.max_steps:
                break
        if env.is_goal(*state):
            successes += 1
            steps_list.append(steps)
    return successes, np.mean(steps_list) if steps_list else None

def main():
    ensure_map()
    grid = load_map_json("maps/campus_map.json")
    env = GridWorld(grid, step_reward=-1, wall_reward=-5, goal_reward=100, max_steps=500, seed=123)
    plot_grid(env.grid, title="Campus Map", savepath="results/map.png")

    print("Running dynamic programming algorithms (Value Iteration, Policy Iteration)...")
    run_dp(env)

    print("Running Q-learning...")
    run_q_learning(env)

    print("Running SARSA...")
    run_sarsa(env)

    print("All done. Check results/ folder for policies and learning curves.")

if __name__ == "__main__":
    main()
