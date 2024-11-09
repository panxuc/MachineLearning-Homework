import numpy as np
import sys
import gymnasium as gym
from typing import List, Tuple


def QLearning(
        env: gym.Env, 
        num_episodes: int = 5000, 
        gamma: float = 0.95, 
        lr: float = 0.1, 
        e: float = 1, 
        decay_rate: float = 0.99
    ) -> Tuple[np.ndarray, List[float]]:
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env:  Environment to compute Q function for. 
    num_episodes:  Number of episodes of training.
    gamma: Discount factor. Number in range [0, 1)
    learning_rate: Learning rate. Number in range [0, 1)
    e: Epsilon value used in the epsilon-greedy method.
    decay_rate: Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    Q: An array of shape [env.observation_space.n x env.action_space.n] representing state, action values
    episode_reward: A list of total rewards of all episode during training
    """
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s, info = env.reset()
        while (True):
            if np.random.rand() > e:
                a = np.argmax(Q[s])
            else:
                a = np.random.randint(env.action_space.n)
            nexts, reward, terminated, turcated, info = env.step(a)
            done = terminated or turcated

            # -------------------------------
            # Your code goes here(1 line)
            Q[s][a] = Q[s][a] + lr * (reward + gamma * np.max(Q[nexts]) - Q[s][a])
            # -------------------------------
            tmp_episode_reward += reward
            s = nexts
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        print("Total reward until episode", i + 1, ":", tmp_episode_reward)
        sys.stdout.flush()
        if i % 10 == 0:
            e = e * decay_rate
    return Q, episode_reward


def Sarsa(
        env: gym.Env, 
        num_episodes: int = 5000, 
        gamma: float = 0.95, 
        lr: float = 0.1, 
        e: float = 1, 
        decay_rate: float = 0.99
    ):
    # num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99
    """Learn state-action values using the Sarsa algorithm with epsilon-greedy exploration strategy.
    Update Q at the end of every episode.

    Parameters
    ----------
    env:  Environment to compute Q function for. 
    num_episodes:  Number of episodes of training.
    gamma: Discount factor. Number in range [0, 1)
    learning_rate: Learning rate. Number in range [0, 1)
    e: Epsilon value used in the epsilon-greedy method.
    decay_rate: Rate at which epsilon falls. Number in range [0, 1)

    Returns
    -------
    Q: An array of shape [env.observation_space.n x env.action_space.n] representing state, action values
    episode_reward: A list of total rewards of all episode during training
    """

    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    episode_reward = np.zeros((num_episodes,))
    for i in range(num_episodes):
        tmp_episode_reward = 0
        s, info = env.reset()
        if np.random.rand() > e:
            a = np.argmax(Q[s])
        else:
            a = np.random.randint(env.action_space.n)
        while True:
            nexts, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            if np.random.rand() > e:
                nexta = np.argmax(Q[nexts])
            else:
                nexta = np.random.randint(env.action_space.n)
            Q[s][a] = Q[s][a] + lr * (reward + gamma * Q[nexts][nexta] - Q[s][a])
            tmp_episode_reward += reward
            s, a = nexts, nexta
            if done:
                break
        episode_reward[i] = tmp_episode_reward
        print(f"Total reward until episode {i + 1}: {tmp_episode_reward}")
        if i % 10 == 0:
            e *= decay_rate
    return Q, episode_reward
