import time
import numpy as np
import gymnasium as gym

def render_single_Q(env:gym.Env, Q):
    """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
    """

    episode_reward = 0
    state, info = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.2)  # Seconds between frames. Modify as you wish.
        action = np.argmax(Q[state])
        state, reward, terminated, turcated, _ = env.step(action)
        done  = terminated or turcated
        episode_reward += reward
    print("Episode reward: %f" % episode_reward)


def evaluate_Q(env, Q, num_episodes=100):
    tot_reward = 0
    for i in range(num_episodes):
        episode_reward = 0
        state, info = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, turcated, _ = env.step(action)
            done  = terminated or turcated
            episode_reward += reward
        tot_reward += episode_reward
    print("Total", tot_reward, "reward in", num_episodes, "episodes")
    print("Average Reward:", tot_reward / num_episodes)
