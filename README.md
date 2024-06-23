# Solving the Mountain Car Problem Using Reinforcement Learning

This repository contains a Jupyter Notebook implementation of the Mountain Car problem using Reinforcement Learning. The Mountain Car problem is a classic control problem in the OpenAI Gym environment.

## Table of Contents

- [Team Members](#team-members)
- [Introduction](#introduction)
- [Environment Initialization](#environment-initialization)
- [Problem Description](#problem-description)
- [Reinforcement Learning Approach](#reinforcement-learning-approach)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Team Members

| Name            | ID          |
|-----------------|-------------|
| Birehan Anteneh | UGR/4886/12 |
| Sefineh Tesfa   | UGR/2844/12 |

## Introduction

The Mountain Car problem is described on the OpenAI Gym website as follows:

> A car is on a one-dimensional track, positioned between two “mountains”. The goal is to drive up the mountain on the right; however, the car’s engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

![Mountain Car](path/to/image)

## Environment Initialization

To begin working with this environment, import and initialize it as follows:

```python
import gym
env = gym.make('MountainCar-v0')
env.reset()
```

## Problem Description

The Mountain Car environment is a challenging problem where the agent (the car) must learn to build enough momentum to reach the top of the mountain. The state space consists of the car's position and velocity, and the action space consists of three possible actions: push left, do nothing, or push right.

## Reinforcement Learning Approach

The approach used in this notebook involves applying Reinforcement Learning techniques to train the agent to solve the Mountain Car problem. The primary focus is on using Q-learning and Deep Q-Networks (DQN).

### Q-Learning

Q-learning is a value-based method of learning how to act optimally in a Markovian environment. It updates the Q-values based on the action taken, the reward received, and the maximum future reward.

### Deep Q-Networks (DQN)

DQN uses a neural network to approximate the Q-values, allowing the agent to learn optimal policies for environments with large state spaces.

## Implementation Details

The notebook includes the following key steps:

1. **Initialization**: Setting up the environment and initializing parameters.
2. **Q-Learning Algorithm**: Implementing the Q-learning algorithm to train the agent.
3. **DQN Implementation**: Using a neural network to approximate Q-values and train the agent using the DQN approach.
4. **Training Loop**: Running episodes to train the agent, updating Q-values or neural network weights.
5. **Evaluation**: Assessing the performance of the trained agent.

## Results

The results section includes graphs and plots showing the performance of the agent over time, including the total reward per episode and the success rate of reaching the top of the mountain.

## How to Run

To run the notebook, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mountain-car-rl.git
    cd mountain-car-rl
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Mountain_Car_Problem_RL.ipynb
    ```
4. Run all cells to train the agent and visualize the results.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- Jupyter Notebook
- NumPy
- Matplotlib
- OpenAI Gym
- TensorFlow or PyTorch (for DQN)

Install the dependencies using:
```bash
pip install -r requirements.txt
```