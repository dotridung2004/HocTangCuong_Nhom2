import numpy as np

def sarsa(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = np.zeros((env.grid.shape[0], env.grid.shape[1], env.n_actions))
    for ep in range(episodes):
        state = env.reset()
        x, y = state
        if np.random.rand() < epsilon:
            action = np.random.randint(env.n_actions)
        else:
            action = np.argmax(Q[x, y])

        done = False
        while not done:
            next_state, reward, done = env.step(action)
            nx, ny = next_state
            if np.random.rand() < epsilon:
                next_action = np.random.randint(env.n_actions)
            else:
                next_action = np.argmax(Q[nx, ny])

            Q[x, y, action] += alpha * (reward + gamma * Q[nx, ny, next_action] - Q[x, y, action])
            state, action = next_state, next_action
            x, y = state
    return Q
