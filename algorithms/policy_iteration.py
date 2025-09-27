import numpy as np

def policy_iteration(env, gamma=0.9, theta=1e-6):
    policy = np.random.randint(env.n_actions, size=(env.grid.shape[0], env.grid.shape[1]))
    V = np.zeros((env.grid.shape[0], env.grid.shape[1]))

    stable = False
    while not stable:
        # Policy Evaluation
        while True:
            delta = 0
            for i in range(env.grid.shape[0]):
                for j in range(env.grid.shape[1]):
                    if (i, j) == env.goal or env.grid[i, j] == 1:
                        continue
                    v = V[i, j]
                    env.state = (i, j)
                    (nx, ny), reward, _ = env.step(policy[i, j])
                    V[i, j] = reward + gamma * V[nx, ny]
                    delta = max(delta, abs(v - V[i, j]))
            if delta < theta:
                break

        # Policy Improvement
        stable = True
        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                if (i, j) == env.goal or env.grid[i, j] == 1:
                    continue
                old_action = policy[i, j]
                values = []
                for action in range(env.n_actions):
                    env.state = (i, j)
                    (nx, ny), reward, _ = env.step(action)
                    values.append(reward + gamma * V[nx, ny])
                policy[i, j] = np.argmax(values)
                if old_action != policy[i, j]:
                    stable = False
    return V, policy
