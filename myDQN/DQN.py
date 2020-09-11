# coding = utf-8
# DQN主要过程
# 记忆库D

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import copy

from myDQN.replay_buffer import ReplayBuffer


class DeepQNetwork:
    def __init__(self, config):
        self.conf = config
        self.n_actions = self.conf['ENV']['n_actions']
        # 经验池，存储S,A,S',R
        self.replay_buffer = ReplayBuffer(self.conf['DQN']['buffer_size'])

        # 强化学习的参数
        self.epsilon_min = self.conf['DQN']['epsilon_min']
        self.epsilon = self.conf['DQN']['init_epsilon']
        self.epsilon_decay = self.conf['DQN']['epsilon_decay']

        self.modelQ = self.create_net()
        self.targetQ = copy.deepcopy(self.modelQ)

    def create_net(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(20, activation='relu'))
        model.add(layers.Dense(15, activation='relu'))
        model.add(layers.Dense(self.n_actions, activation='relu'))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.conf["DQN"]["learning_rate"]),
            loss=[tf.keras.losses.mean_squared_error]
        )
        return model

    # 贪心算法根据状态选择动作
    def choose_action(self, observation):
        if random.random() < self.epsilon:
            action = random.randint(0, self.n_actions-1)
        else:
            # modelQ预测的是一个observition的Q值
            action = np.argmax(self.modelQ.predict(np.expand_dims(np.array(observation), axis=0))[0])
        return action

    def learn(self, name="Nature"):

        batch_data = self.replay_buffer.sample(self.conf['DQN']['batch_size'])
        observation, action, observation_, reward, done = zip(*batch_data)
        # 如果是 DQN
        if name == "DQN":
            next_q_value = self.targetQ.predict(np.array(observation_))
            max_next_q = np.max(next_q_value, axis=1)
        # 如果是Nature DQN
        if name == "Nature":
            next_q_value = self.targetQ.predict(np.array(observation_))
            max_next_q = np.max(next_q_value, axis=1)
        # 如果是Double DQN
        elif name == "Double":
            next_q_value = self.modelQ.predict(np.array(observation_))
            action_ = tf.one_hot(np.argmax(next_q_value, axis=1), depth=self.n_actions)
            max_next_q = np.max(action_ * self.targetQ(np.array((observation_))), axis=1)

        target_y = reward + (1-np.array(done)) * self.conf['DQN']['gamma'] * max_next_q
        target_q = self.modelQ.predict(np.array(observation))
        for k in range(len(batch_data)):
            target_q[k][action[k]] = target_y[k]
        self.modelQ.train_on_batch(np.array(observation), target_q)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_targetQ(self):
        q_weights = self.modelQ.get_weights()
        q_target_weights = self.targetQ.get_weights()

        tau = self.conf['DQN']['tau']
        q_weights = [tau * w for w in q_weights]
        q_target_weights = [(1. - tau) * w for w in q_target_weights]
        new_weights = [
            q_weights[i] + q_target_weights[i]
            for i in range(len(q_weights))
        ]
        self.targetQ.set_weights(new_weights)




