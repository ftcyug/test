# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 09:11:12 2021

@author: 15091
"""
import torch
import torch.nn as nn
from   torch.autograd import Variable
import numpy as np
import gym

batch_size=32
lr=0.01
epsilon=0.9
gamma=0.9
target_replace_iter=100
memory_capacity=2000
env=gym.make('CartPole-V0')
env=env.unwrapped
n_actions=env.acion_space.n
n_states=env.observation_space.shape[0]
print('number of actions are:{}'.format(n_actions))
print('number of states are:{}'.format(n_states))
#%%
class q_net(nn.Module):
    def __init__(self,hidden=50):
        super(q_net,self).__init__()
        self.fc=nn.Sequential(
                nn.Linear(n_states,hidden),
                nn.ReLU(True),
                nn.Linear(hidden,n_actions)
                )
        nn.init.normal(self.fc[0].weight,std=0.1)
        nn.init.normal(self.fc[2].weight,std=0.1)
    def forward(self, x):
        actions_value = self.fc(x)  # x是什么意思？有什么作用
        return actions_value
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = q_net(), q_net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
    def choose_action(self, s):
        s = torch.tensor(torch.unsqueeze(torch.FloatTensor(s), 0))
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net(s)
            action = torch.max(actions_value, 1)[1].data[0]
        else:
            action = np.random.randint(0, n_actions)
        return action
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    def learn(self):
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.tensor(torch.FloatTensor(b_memory[:, :n_states]))
        b_a = torch.tensor(
            torch.LongTensor(b_memory[:, n_states:n_states + 1].astype(int))
        )
        b_r = torch.tensor(
            torch.FloatTensor(b_memory[:, n_states + 1:n_states + 2])
        )
        b_s_ = torch.tensor(torch.FloatTensor(b_memory[:, -n_states:]))
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(
            b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.criterion(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
dqn_trainer = DQN()
print('collecting experience...')
all_reward = []
for i_episode in range(300):
    s = env.reset()
    reward = 0
    while True:
        env.render()
        a = dqn_trainer.choose_action(s)
        s_, r, done, info = env.step(a)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        dqn_trainer.store_transition(s, a, r, s_)
        reward += r
        if dqn_trainer.memory_counter > memory_capacity:
            dqn_trainer.learn()
            if done:
                print(i_episode, round(reward, 3))
                all_reward.append(reward)
                break
        if done:
            break
        s = s_