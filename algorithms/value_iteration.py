import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    V = np.zeros((env.grid.shape[0], env.grid.shape[1]))
    policy = np.zeros_like(V, dtype=int)

    while True:
        delta = 0
        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                if (i, j) == env.goal or env.grid[i, j] == 1:
                    continue
                v = V[i, j]
                values = []
                for action in range(env.n_actions):
                    env.state = (i, j)
                    (nx, ny), reward, _ = env.step(action)
                    values.append(reward + gamma * V[nx, ny])
                V[i, j] = max(values)
                policy[i, j] = np.argmax(values)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V, policy
