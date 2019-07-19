import numpy as np
import torch.nn as nn
import gym
import torch
import torch.nn.functional as F
import random
import math
import matplotlib.pyplot as plt


class PolicyGradient(nn.Module):

    def __init__(self):
        super(PolicyGradient, self).__init__()
        self.linear = nn.Linear(4, 8)
        self.linear2 = nn.Linear(8, 2)

    def forward(self, state):
        input1 = torch.autograd.Variable(torch.FloatTensor([state]))
        return F.log_softmax(self.linear2(F.sigmoid(self.linear(input1))))


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.linear = nn.Linear(4, 8)
        self.linear2 = nn.Linear(2, 4)
        self.linear3 = nn.Linear(12, 1)

    def forward(self, state, action):
        x = torch.tensor([state], dtype=torch.float32)
        x_state = F.sigmoid(self.linear(x))

        # import ipdb; ipdb.set_trace()
        x_action = F.sigmoid(self.linear2(action.unsqueeze(0)))
        x = torch.cat([x_state, x_action], dim=1)
        x = self.linear3(x)
        return x


def train():

    env = gym.make('CartPole-v0')

    env.seed(1)

    EPISODES = 5000

    criterion = torch.nn.NLLLoss()
    critic_loss = torch.nn.MSELoss()
    policy = PolicyGradient()
    critic = Critic()
    optimizer = torch.optim.RMSprop(policy.parameters(), lr=0.002)
    optimizer_critic = torch.optim.RMSprop(critic.parameters(), lr=0.002)
    env.reset()
    decay = 1
    resultlist = []
    for episode in range(EPISODES):
        observations = []
        observation = env.reset()
        while True:
            action = int(
                np.random.choice(range(2),
                p=np.exp(policy(observation).data.numpy()[0]))
            )
            observation_, reward, finished, info = env.step(action)
            observations.append((observation, action, reward, observation_))
            observation = observation_
            if finished:
                rewardlist = [x[2] for x in observations]
                cumulative = 0
                savelist = []
                for rew in rewardlist[::-1]:
                    cumulative = cumulative * decay + rew/200
                    savelist.append(cumulative)
                savelist = savelist[::-1]
                resultlist.append(savelist[0])
                if episode % ((EPISODES // 10) - 1) == 0:
                    plt.plot(resultlist)
                    plt.show()
                savelist = np.array(savelist)

                for (observation, action, reward, next_observation), cum_reward in zip(observations, savelist):

                    onehot_action = torch.zeros(2)
                    onehot_action[action] = 1

                    crit_score = critic(observation, onehot_action)

                    loss = critic_loss(
                        crit_score,
                        torch.autograd.Variable(
                            torch.FloatTensor([cum_reward])
                        ).view(1, 1)
                    )
                    loss.backward()
                    optimizer.zero_grad()
                    optimizer_critic.step()
                    optimizer_critic.zero_grad()
                    crit_score = float(
                        critic(observation, onehot_action).data.numpy()[0][0]
                    )
                    action = torch.autograd.Variable(
                        torch.LongTensor([action])
                    )
                    result = policy(observation)
                    loss = criterion(result, action)
                    (loss * crit_score).backward()
                    optimizer.step()
                    optimizer.zero_grad()
                break
