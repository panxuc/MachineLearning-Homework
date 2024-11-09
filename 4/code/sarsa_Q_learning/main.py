import gymnasium as gym
import matplotlib.pyplot as plt
from algorithms import QLearning, Sarsa
from utils import render_single_Q, evaluate_Q
import random
import numpy as np


class FrozenLakeRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, terminated, turcated, info =  super().step(action)
        reward = -1.0 if terminated and not reward else reward # set reward to -1 when falling into a hole
        reward = 2.0 if reward == 1.0 else reward # set reward to 2 when reach the goal
        reward = -0.03 if reward == 0.0 else reward # add a -0.03 penalty each step
        # reward = -0.3 if reward == 0.0 else reward # add a -0.3 penalty each step
        return obs, reward, terminated, turcated, info

def smooth(data:np.ndarray, window_size=70):
    return np.convolve(data, np.ones(window_size, dtype=int), 'valid') / window_size

# Feel free to run your own debug code in main!
def main():
    num_episodes = 7000
    seed = 0
    
    #######=========================######
    # Do NOT modify
    map_name = "4x4"
    def make_env(is_render=False):
        return FrozenLakeRewardWrapper(
            gym.make('FrozenLake-v1', 
                     desc=None, map_name=map_name, is_slippery=True, render_mode="human" if is_render else None))
    env, render_env = make_env(), make_env(is_render=True)
    #######=========================######
    lr = 0.001
    # q_learning
    Q1, Q_rewards1 = QLearning(env, num_episodes, lr=lr)
    # sarsa
    Q2, Q_rewards2 = Sarsa(env, num_episodes, lr=lr)
    
    render_single_Q(render_env, Q1)
    render_single_Q(render_env, Q2)
    
    evaluate_Q(env, Q1, 200)
    print([int(np.argmax(i)) for i in Q1])
    
    evaluate_Q(env, Q2, 200)
    print([int(np.argmax(i)) for i in Q2])
    
    Q_rewards1  = smooth(Q_rewards1)
    Q_rewards2  = smooth(Q_rewards2)
    
    # plt.plot(range(len(Q_rewards1)), Q_rewards1)

    # Plot the learning curves of two methods
    plt.plot(range(len(Q_rewards1)), Q_rewards1, alpha=0.7, label="Q-learning")
    plt.plot(range(len(Q_rewards2)), Q_rewards2, alpha=0.7, label="Sarsa")
    plt.xlabel("Episodes")
    plt.ylabel("Total rewards")
    plt.legend()
    plt.title(f"Learning Curve with lr={lr}")
    # plt.show()
    plt.savefig(f"./output/{lr}_1.svg")
    # plt.savefig(f"./output/{lr}_2.svg")


if __name__ == '__main__':
    main()
