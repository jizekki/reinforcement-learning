import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gym
from gym.spaces import Discrete, Box
import random

env = gym.make("CartPole-v0")
obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

# Constants for training
learning_rate = 1e-2
epochs = 5000
batch_size = 500
epsilon = 0.1
##########################

#############################################
####### BUILDING A NEURAL NETWORK ###########
##### REPRESENTING ACTION STATE VALUES ######
#############################################

# net_Qvalue is a neural network representing an action state value function:
# it takes as inputs observations and outputs values for each action
net_Qvalue = nn.Sequential(
    nn.Linear(obs_dim, 32),
    nn.Tanh(),
    nn.Linear(32, n_acts)
)

# net_Qvalue_target is another one
net_Qvalue_target = nn.Sequential(
    nn.Linear(obs_dim, 32),
    nn.Tanh(),
    nn.Linear(32, n_acts)
)
net_Qvalue_target.eval()


def choose_action(observation):
    if random.random() < epsilon:
        return random.randrange(n_acts)
    else:
        with torch.no_grad():
            q_values = net_Qvalue(observation)
            return q_values.max(0)[1].item()


def compute_loss(batch_observations, batch_actions, batch_rewards, batch_next_observations, batch_non_final):
    batch_q_values = net_Qvalue(batch_observations)
    # print("Qmean:", torch.mean(batch_q_values.detach()))
    batch_q_value = batch_q_values.gather(
        1, batch_actions.unsqueeze(1)).squeeze(1)

    batch_q_value_next = torch.zeros_like(batch_q_value)
    with torch.no_grad():
        next_non_final_observations = batch_next_observations[batch_non_final]
        batch_q_values_next = net_Qvalue(next_non_final_observations)
        _, batch_max_indices = batch_q_values_next.max(dim=1)
        batch_q_values_next = net_Qvalue_target(next_non_final_observations)
        batch_q_value_next[batch_non_final] = batch_q_values_next.gather(
            1, batch_max_indices.unsqueeze(1)).squeeze(1)

    batch_expected_q_value = batch_rewards + batch_q_value_next
    loss = (batch_q_value - batch_expected_q_value).pow(2).mean()
    return loss


# make optimizer
optimizer = Adam(net_Qvalue.parameters(), lr=learning_rate)


def DQN():
    for i in range(epochs):
        # we copy the parameters of Qvalue into Qvalue_target every 10 iterations
        if i % 10 == 0:
            net_Qvalue_target.load_state_dict(net_Qvalue.state_dict())

        batch_observations = []
        batch_actions = []
        batch_rewards = []
        batch_next_observations = []
        batch_non_final = []

        total_rewards = []

        # for statistics over all episodes run in the first step
        episodes = 0
        total_reward = 0
        next_batch_index = 0

        # reset episode-specific variables
        observation = env.reset()
        done = False

        # First step: collect experience by simulating the environment using the current policy
        while next_batch_index < batch_size:
            action = choose_action(torch.as_tensor(
                observation, dtype=torch.float32))
            batch_observations.append(observation)
            observation, reward, done, _ = env.step(action)
            batch_next_observations.append(observation)
            batch_actions.append(action)
            batch_rewards.append(reward)
            total_reward += reward
            batch_non_final.append(not done)
            if done:
                observation = env.reset()
                episodes += 1
                total_rewards.append(total_reward)
                total_reward = 0
            next_batch_index += 1

        # Second step: update the policy
        # we take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(torch.as_tensor(batch_observations, dtype=torch.float32),
                                  torch.as_tensor(
                                      batch_actions, dtype=torch.int64),
                                  torch.as_tensor(
                                      batch_rewards, dtype=torch.float32),
                                  torch.as_tensor(
                                      batch_next_observations, dtype=torch.float32),
                                  torch.as_tensor(
                                      batch_non_final, dtype=torch.bool)
                                  )
        batch_loss.backward()
        optimizer.step()

        if i % 20 == 0:
            print('epoch: %3d \t loss: %.3f \t mean_total_rewards: %.3f' %
                  (i, batch_loss, np.mean(total_rewards)))


DQN()

###### EVALUATION ############


def run_episode(env, render=False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


epsilon = 0
policy_scores = [run_episode(env) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

for _ in range(2):
    run_episode(env, True)

env.close()
