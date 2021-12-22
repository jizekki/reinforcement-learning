from functools import total_ordering
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make("CartPole-v0")

env.seed(0)
env.action_space.seed(0)
np.random.seed(0)

NB_STATES = 162  # 3 * 3 * 6 * 3


def obs_to_state(obv):
    x, x_dot, theta, theta_dot = obv

    # X_pos Pass
    if x < -.8:
        state = 0
    elif x < .8:
        state = 1
    else:
        state = 2

    # X_velocity Pass
    if x_dot < -.5:
        pass
    elif x_dot < .5:
        state += 3
    else:
        state += 6

    if theta < np.radians(-12):
        pass
    elif theta < np.radians(-1.5):
        state += 9
    elif theta < np.radians(0):  # goldzone
        state += 18
    elif theta < np.radians(1.5):
        state += 27
    elif theta < np.radians(12):
        state += 36
    else:
        state += 45

    if theta_dot < np.radians(-50):
        pass
    elif theta_dot < np.radians(50):
        state += 54
    else:
        state += 108

    return state


MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1


def update_explore_rate(episode):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode + 1) / 25)))


def update_learning_rate(episode):
    return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 50))))


def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            state = obs_to_state(obs)
            action = policy[state]

        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


#run_episode(env, render=False)

EPISODES = 2000
TIMEOUT = 250

GAMMA = 1


def Q_learning():
    Qvalue = np.random.rand(NB_STATES, env.action_space.n)
    for episode in range(EPISODES):
        obs = env.reset()
        lr = update_learning_rate(episode)
        eps = update_explore_rate(episode)
        state = obs_to_state(obs)
        done = False
        total_reward = 0
        while(not done):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qvalue[state])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            new_state = obs_to_state(obs)
            Qvalue[state, action] += lr * \
                (reward + GAMMA *
                 np.max(Qvalue[new_state]) - Qvalue[state, action])
            state = new_state
        if episode % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' %
                  (episode+1, total_reward))
    return Qvalue


def SARSA():
    Qvalue = np.random.rand(NB_STATES, env.action_space.n)
    for episode in range(EPISODES):
        obs = env.reset()
        lr = update_learning_rate(episode)
        eps = update_explore_rate(episode)

        state = obs_to_state(obs)
        done = False
        total_reward = 0
        while(not done):
            if np.random.random() < eps:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qvalue[state])
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            new_state = obs_to_state(obs)
            if np.random.random() < eps:
                action_greedy = env.action_space.sample()
            else:
                action_greedy = np.argmax(Qvalue[new_state])
            Qvalue[state, action] += lr * \
                (reward + GAMMA * Qvalue[new_state,
                                         action_greedy] - Qvalue[state, action])
            state = new_state

        if episode % 200 == 0:
            print('Iteration #%d -- Total reward = %d.' %
                  (episode+1, total_reward))
    return Qvalue


Qvalue = Q_learning()
#Qvalue = SARSA()
policy = np.argmax(Qvalue, axis=1)

policy_scores = [run_episode(env, policy, render=False) for _ in range(100)]
print("Average score of the policy: ", np.mean(policy_scores))

# for _ in range(2):
#   run_episode(env, policy, True)
