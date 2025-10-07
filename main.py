# file: main.py
import os
import argparse
import numpy as np
from env.gridworld import GridWorld
from utils.map_io import load_map_json, save_map_json
from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from algorithms.q_learning import q_learning, state_to_key
from algorithms.sarsa import sarsa, state_to_key as sarsa_state_to_key
from utils.visualize import plot_grid, draw_policy, draw_optimal_path, plot_learning_curve

CONFIG = {
    "dp": {"gamma": 0.9},
    "q_learning": {
        "num_episodes": 15000, 
        "alpha": 0.1, 
        "gamma": 0.9, 
        "epsilon": 1.0, 
        "epsilon_decay": 0.9995, 
        "seed": 42
    },
    "sarsa": {
        "num_episodes": 15000,       
        "alpha": 0.05,              
        "gamma": 0.99,              
        "epsilon": 1.0, 
        "epsilon_decay": 0.9999,    
        "seed": 24
    },
    "env": {
        "step_reward": -1, 
        "wall_reward": -5, 
        "goal_reward": 100, 
        "max_steps": 1000, 
        "seed": 123
    }
}

def choose_start_goal(grid, save_path):
    """Hàm nhập Start/Goal an toàn"""
    rows, cols = grid.shape
    print(f"\nBản đồ có kích thước: {rows} x {cols}")
    print("⚠ Lưu ý: chỉ số bắt đầu từ 0")

    def get_coord(point_name):
        while True:
            try:
                r = int(input(f"Nhập hàng (row) điểm {point_name}: "))
                c = int(input(f"Nhập cột (col) điểm {point_name}: "))
                
                if not (0 <= r < rows and 0 <= c < cols):
                    print(f"❌ Lỗi: Tọa độ nằm ngoài bản đồ. Vui lòng nhập lại.")
                    continue
                
                if grid[r, c] == 0:
                    print(f"❌ Lỗi: Không thể đặt điểm trên tường. Vui lòng chọn ô khác.")
                    continue
                return r, c
            except ValueError:
                print("❌ Lỗi: Bạn phải nhập số nguyên.")
    
    print("\n--- Chọn điểm xuất phát (START) ---")
    sr, sc = get_coord("START")
    print("\n--- Chọn điểm đích (GOAL) ---")
    gr, gc = get_coord("GOAL")

    grid[grid == 2] = 1 
    grid[grid == 3] = 1
    grid[sr, sc] = 2
    grid[gr, gc] = 3
    
    save_map_json(grid, save_path)
    print(f"\n✅ Đã cập nhật: Start=({sr},{sc}), Goal=({gr},{gc}) và lưu vào '{save_path}'")
    return grid, (sr, sc)

def convert_policy_keys(Q):
    """Chuyển từ state_to_key (tuple of tuple) sang (r,c) để vẽ policy"""
    policy_rc = {}
    for state_key, arr in Q.items():
        grid_array = np.array(state_key)
        pos = np.argwhere(grid_array == 4)
        if len(pos) == 0:
            continue
        r, c = pos[0]
        policy_rc[(r, c)] = int(np.argmax(arr))
    return policy_rc

def main(args):
    os.makedirs("results", exist_ok=True)
    os.makedirs("maps", exist_ok=True)

    try:
        grid = load_map_json(args.map)
    except FileNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy file map '{args.map}'.")
        return

    grid, start_pos = choose_start_goal(grid, save_path=args.map)
    env = GridWorld(grid, **CONFIG['env'])

    plot_grid(env.grid, title=f"Bản đồ: {os.path.basename(args.map)}", savepath="results/map.png")
    
    all_results = {}
    
    print("\n▶ Đang chạy Value Iteration...")
    _, pi_vi = value_iteration(env, **CONFIG['dp'])
    all_results["Value Iteration"] = {"policy": pi_vi}
    print("\n▶ Đang chạy Policy Iteration...")
    _, pi_pi = policy_iteration(env, **CONFIG['dp'])
    all_results["Policy Iteration"] = {"policy": pi_pi}

    print("\n▶ Đang chạy Q-learning...")
    Q_ql, rewards_ql, _ = q_learning(env, **CONFIG['q_learning'])
    policy_ql = convert_policy_keys(Q_ql)
    all_results["Q-learning"] = {"policy": policy_ql, "rewards": rewards_ql}
    
    print("\n▶ Đang chạy SARSA...")
    Q_sarsa, rewards_sarsa, _ = sarsa(env, **CONFIG['sarsa'])
    policy_sarsa = convert_policy_keys(Q_sarsa)
    all_results["SARSA"] = {"policy": policy_sarsa, "rewards": rewards_sarsa}

    print("\n" + "="*50)
    print("🎨 ĐANG VẼ KẾT QUẢ CHO TỪNG THUẬT TOÁN 🎨")
    for name, result in all_results.items():
        print(f"\n--- Kết quả cho: {name} ---")
        policy = result["policy"]
        
        draw_policy(
            env.grid, 
            policy, 
            title=f"Full Policy - {name}",
            savepath=f"results/full_policy_{name.lower().replace(' ', '_')}.png"
        )
        
        draw_optimal_path(
            env.grid, 
            policy, 
            start_pos, 
            title=f"Optimal Path - {name}",
            savepath=f"results/optimal_path_{name.lower().replace(' ', '_')}.png"
        )
        
        if "rewards" in result:
            plot_learning_curve(
                result["rewards"], 
                title=f"{name} Rewards",
                savepath=f"results/learning_{name.lower()}.png"
            )
            
    print("="*50)
    print("✅ Hoàn thành!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chạy các thuật toán RL trên GridWorld.")
    parser.add_argument("--map", type=str, default="maps/dhtl_map.json", help="Đường dẫn tới file map JSON.")
    args = parser.parse_args()
    main(args)
