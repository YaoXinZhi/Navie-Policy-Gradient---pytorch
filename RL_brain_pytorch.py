#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Created on 09/10/2019 9:36 
@Author: XinZhi Yao 
"""

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd.variable as Variable


"""
code of Policy Gradient
"""
np.random.seed(26)
torch.manual_seed(26)

class PolicyGradient(nn.Module):
    def __init__(self, n_action, n_features, batch_size=10, learning_rate=0.01, reward_decay=0.95, out_graph=False):
        super(PolicyGradient, self).__init__()
        self.batch_size = batch_size
        self.n_actions = n_action
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # experience observation, experience actions, experience rewards
        # save experience
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()


    def _build_net(self):

        # self.obs = torch.nn.Parameter(torch.Tensor(self.batch_size, self.n_features))
        # self.obs = torch.Tensor()
        # self.acts = torch.Tensor()
        # self.vt = torch.Tensor()

        self.net = nn.Sequential(
            nn.Linear(self.n_features, 10),
            nn.Tanh(),
            nn.Linear(10, self.n_actions),
        )

        self.init_weight()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

    def init_weight(self):
        for weight in self.parameters():
            init.normal_(weight, mean=0, std=0.3)

    def choose_action(self, observation):
        # 需要改一下
        observation = torch.from_numpy(observation).unsqueeze(0).float()
        prob_weight = F.softmax(self.net(observation), 1)
        p_choose = np.round(prob_weight.view(-1).detach().numpy(), 3)
        action = np.random.choice(range(prob_weight.size(1)), p=p_choose)
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(RL):
        # discount and normalize episode reward
        discounted_ep_rs_norm = RL._discount_and_norm_rewards()

        tensor_obs = torch.from_numpy(np.array(RL.ep_obs)).float()
        tensor_acts = torch.LongTensor(RL.ep_as).unsqueeze(1)
        tensor_vt = torch.FloatTensor(discounted_ep_rs_norm)


        # train on episode
        RL.all_act_prob = F.softmax(RL.net(tensor_obs), dim=1)
        selected_action_prob = torch.gather(RL.all_act_prob, 1, tensor_acts)
        # action_one_hot = torch.zeros(self.all_act_prob.shape).scatter_(dim=1, index=self.ep_as, src=1)

        neg_log_prob = torch.sum(-torch.log(selected_action_prob))
        loss = torch.mean(torch.mul(neg_log_prob, tensor_vt))

        RL.optimizer.zero_grad()
        loss.backward()
        RL.optimizer.step()

        RL.ep_obs, RL.ep_as, RL.ep_rs = [], [], []
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
