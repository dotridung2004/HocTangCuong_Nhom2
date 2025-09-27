import numpy as np

def q_learning(env, episodes=500, alpha=0.1, gamma=0.9, epsilon=0.2):
    Q = np.zeros((env.grid.shape[0], env.grid.shape[1], env.n_actions))
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            x, y = state
            # ε-greedy
            if np.random.rand() < epsilon:
                action = np.random.randint(env.n_actions)
            else:
                action = np.argmax(Q[x, y])

            next_state, reward, done = env.step(action)
            nx, ny = next_state

            Q[x, y, action] += alpha * (reward + gamma * np.max(Q[nx, ny]) - Q[x, y, action])
            state = next_state
    return Q
