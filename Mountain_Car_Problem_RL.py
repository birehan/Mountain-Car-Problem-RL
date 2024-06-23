import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

# Initialize environment
def initialize_environment():
    env = gym.make('MountainCar-v0')
    return env

# Define Q-Learning agent
class QLearningAgent:
    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state, alpha, gamma):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + gamma * self.q_table[next_state][best_next_action]
        self.q_table[state][action] += alpha * (td_target - self.q_table[state][action])

# Define DQN agent
class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_shape=self.state_shape),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_space, activation='linear')
        ])
        model.compile(optimizer=optimizers.Adam(), loss='mse')
        return model

    def choose_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.choice(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def train(self, memory, batch_size, gamma):
        if len(memory) < batch_size:
            return
        batch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

# Training function
def train_agent(env, agent, episodes, alpha, gamma, epsilon, epsilon_decay):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        done = False
        while not done:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
            agent.update_q_table(state, action, reward, next_state, alpha, gamma)
            state = next_state
            total_reward += reward
            if done:
                rewards.append(total_reward)
                epsilon *= epsilon_decay
                print(f"Episode: {episode}, Reward: {total_reward}, Epsilon: {epsilon}")
    return rewards

# Plot results
def plot_rewards(rewards):
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()

# Main function
if __name__ == "__main__":
    env = initialize_environment()
    state_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = QLearningAgent(state_space, action_space)
    episodes = 1000
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    rewards = train_agent(env, agent, episodes, alpha, gamma, epsilon, epsilon_decay)
    plot_rewards(rewards)
