import gymnasium as gym
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.distributions import Categorical
import torch.nn as nn
import numpy as np
import random

# see https://gymnasium.farama.org/environments/classic_control/cart_pole/
# understand environment, state, action and other definitions first before your dive in.

ENV_NAME = 'CartPole-v1'

# Hyper Parameters
# Following params work well if your implement Policy Gradient correctly.
# You can also change these params.
EPISODE = 3000  # total training episodes
STEP = 5000  # step limitation in an episode
EVAL_EVERY = 10  # evaluation interval
TEST_NUM = 5  # number of tests every evaluation
GAMMA = 0.95  # discount factor
LEARNING_RATE = 3e-3  # learning rate for mlp and ac


# A simple mlp implemented by PyTorch #
# it receives (N, D_in) shaped torch arrays, where N: the batch size, D_in: input state dimension
# and outputs the possibility distribution for each action and each sample, shaped (N, D_out)
# e.g.
# state = torch.randn(10, 4)
# outputs = mlp(state)  #  output shape is (10, 2) in CartPole-v0 Game
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


class AC(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_pi = nn.Linear(hidden_dim, output_dim)
        self.fc_v = nn.Linear(hidden_dim, 1)

    def pi(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        return x

    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v


class REINFORCE:
    def __init__(self, env):
        # init parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        self.last_state = None
        self.net = MLP(input_dim=self.state_dim, output_dim=self.action_dim)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def predict(self, observation, deterministic=False):
        observation = torch.FloatTensor(observation).unsqueeze(0)
        action_score = self.net(observation)
        probs = F.softmax(action_score, dim=1)
        m = Categorical(probs)
        if deterministic:
            action = torch.argmax(probs, dim=1)
        else:
            action = m.sample()
        return action, probs

    def store_transition(self, s, a, p, r):
        self.states.append(s)
        self.actions.append(a)
        self.action_probs.append(p)
        self.rewards.append(r)

    def learn(self):
        # Please make sure all variables used to calculate loss are of type torch.Tensor, or autograd may not work properly.
        # You need to calculate the loss of each step of the episode and store them in '''loss'''.
        # The variables you should use are: self.rewards, self.action_probs, self.actions.
        # self.rewards=[R_1, R_2, ...,R_T], self.actions=[A_0, A_1, ...,A_(T-1)]
        # self.action_probs corresponds to the probability of different actions of each timestep, see predict() for details

        loss = []
        # -------------------------------
        # Your code goes here
        # TODO Calculate the loss of each step of the episode and store them in '''loss'''
        G = []
        cumulative = 0
        for reward in reversed(self.rewards):
            cumulative = reward + GAMMA * cumulative
            G.insert(0, cumulative)
        G = torch.tensor(G, dtype=torch.float32)
        G = (G - G.mean()) / (G.std() + 1e-8)
        for action_prob, action, reward in zip(self.action_probs, self.actions, G):
            log_prob = torch.log(action_prob.squeeze(0)[action])
            loss.append(-log_prob * reward)
        # -------------------------------

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()


class TDActorCritic(REINFORCE):
    def __init__(self, env):
        super().__init__(env)
        self.ac = AC(input_dim=self.state_dim, output_dim=self.action_dim)
        # override
        self.net = self.ac.pi
        self.done = None
        self.optim = torch.optim.Adam(self.ac.parameters(), lr=LEARNING_RATE)

    def make_batch(self):
        done_lst = [1.0 if i != len(self.states) - 1 else 0.0 for i in range(len(self.states))]

        self.last_state = torch.tensor(self.last_state, dtype=torch.float).reshape(1, -1)
        self.states = torch.tensor(np.array(self.states), dtype=torch.float)
        self.done = torch.tensor(done_lst, dtype=torch.float).reshape(-1, 1)
        self.actions = torch.tensor(self.actions, dtype=torch.int64).reshape(-1, 1)
        self.action_probs = torch.cat(self.action_probs)
        self.states_prime = torch.cat((self.states[1:], self.last_state))
        self.rewards = torch.tensor(self.rewards, dtype=torch.float).reshape(-1, 1) / 100.0

    def learn(self):
        # Please make sure all variables are of type torch.Tensor, or autograd may not work properly.
        # You only need to calculate the policy loss.
        # The variables you should use are: self.rewards, self.action_probs, self.actions, self.states_prime, self.states.
        # self.states=[S_0, S_1, ...,S_(T-1)], self.states_prime=[S_1, S_2, ...,S_T], self.done=[1, 1, ..., 1, 0]
        # Invoking self.ac.v(self.states) gives you [v(S_0), v(S_1), ..., v(S_(T-1))]
        # For the final timestep T, delta_T = R_T - v(S_(T-1)), v(S_T) = 0
        # You need to use .detach() to stop delta's gradient in calculating policy_loss, see value_loss for an example

        policy_loss = None
        td_target = None
        delta = None
        self.make_batch()
        # -------------------------------
        # Your code goes here
        # TODO Calculate policy_loss
        v_s = self.ac.v(self.states)
        v_s_prime = self.ac.v(self.states_prime) * self.done
        td_target = self.rewards + GAMMA * v_s_prime
        delta = td_target - v_s
        action_log_probs = torch.log(self.action_probs.gather(1, self.actions))
        policy_loss = -(action_log_probs * delta.detach()).mean()
        # -------------------------------

        # compute value loss and total loss
        # td_target is used as a scalar here, and is detached to stop gradient
        value_loss = F.smooth_l1_loss(self.ac.v(self.states), td_target.detach())
        loss = policy_loss + value_loss

        # code for autograd and back propagation
        self.optim.zero_grad()
        loss = loss.mean()
        loss.backward()
        self.optim.step()

        self.states, self.actions, self.action_probs, self.rewards = [], [], [], []
        return loss.item()


def main():
    # initialize OpenAI Gym env and PG agent
    SEED = [0, 1437, 114514]

    for seed in SEED:

        losses = []
        loss_episodes = []
        rewards = []
        reward_episodes = []

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_env = gym.make(ENV_NAME)
        render_env = gym.make(ENV_NAME, render_mode="human")
        RENDER = False

        # You may uncomment the line below to enable rendering for visualization.
        # RENDER = True

        # uncomment to switch between methods
        # agent = REINFORCE(train_env)
        agent = TDActorCritic(train_env)

        for episode in range(EPISODE):
            # initialize task
            env = train_env
            state, _ = env.reset(seed=random.randint(0,int(1e10)))
            agent.last_state = state
            # Train
            for step in range(STEP):
                action, probs = agent.predict(state)
                next_state, reward, terminated, turcated, _ = env.step(action.item())
                done = terminated or turcated
                agent.store_transition(state, action, probs, reward)
                state = next_state
                if done:
                    loss = agent.learn()
                    losses.append(loss)
                    loss_episodes.append(episode)
                    break

            # Test
            env = render_env if RENDER else train_env
            if episode % EVAL_EVERY == 0:
                total_reward = 0
                for i in range(TEST_NUM):
                    state, _ = env.reset(seed=random.randint(0,int(1e10)))
                    for j in range(STEP):
                        action, _ = agent.predict(state, deterministic=True)
                        next_state, reward, terminated, turcated, _ = env.step(action.item())
                        done = terminated or turcated
                        total_reward += reward
                        state = next_state
                        if done:
                            break
                avg_reward = total_reward / TEST_NUM
                rewards.append(avg_reward)
                reward_episodes.append(episode)

                # Your avg_reward should reach 200(cartpole-v0)/500(cartpole-v1) after a number of episodes.
                print('episode: ', episode, 'Evaluation Average Reward:', avg_reward)

        plt.figure()
        plt.plot(loss_episodes, losses)
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title(f'Loss curve, seed={seed}')
        plt.savefig(f'./output/{agent.__class__.__name__}_loss_{seed}.svg')

        plt.figure()
        plt.plot(reward_episodes, rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward curve, seed={seed}')
        plt.savefig(f'./output/{agent.__class__.__name__}_reward_{seed}.svg')


if __name__ == '__main__':
    main()
