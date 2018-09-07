import numpy as np
from collections import deque
import random

from critic_net import CriticNet
from actor_net import ActorNet

# parameters
REPLAY_BUFFER_SIZE = 10000
MINI_BATCH_SIZE = 128
GAMMA = 0.99

class DDPGAgent:
    def __init__(self, num_states, num_actions, action_max, action_min):
        """

        :param states: list
        :param actions: list
        :param action_max: list
        :param action_min: list
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.action_max = action_max
        self.action_min = action_min

        self.replay_buffer = deque()

        self.critic_net = CriticNet(self.num_states, self.num_actions, self.action_max, self.action_min)
        self.actor_net = ActorNet(self.num_states, self.num_actions, self.action_max)

    def feed_forward_actor(self, state):
        return self.actor_net.forward_learned_net(state)

    # メモリ（サンプルバッチ部分）
    def add_experience(self, s_t, a_t, s_t_1, reward, done):
        self.replay_buffer.append((s_t, a_t, s_t_1, reward, done))

        if (len(self.replay_buffer) > REPLAY_BUFFER_SIZE):
            self.replay_buffer.popleft()

    def train(self):
        # Prepare minibatch sampling from replay buffer
        # ランダムにバッチを取りだす
        batch = random.sample(self.replay_buffer, MINI_BATCH_SIZE)
        # 各変数にバッチの情報を配列で渡す
        s_batch = np.array([item[0] for item in batch])
        a_batch = np.array([item[1] for item in batch])
        s_1_batch = np.array([item[2] for item in batch])
        reward_batch = np.array([[item[3]] for item in batch])
        done_batch = np.array([item[4] for item in batch])

        # Train learned critic network
        q_t_1 = self.critic_net.forward_target_net(s_1_batch, self.actor_net.forward_target_net(s_1_batch))
        target_q_batch = []
        for i in range(MINI_BATCH_SIZE):
            if done_batch[i]:
                target_q_batch.append(reward_batch[i][0])
            else:
                target_q_batch.append(reward_batch[i][0] + GAMMA * q_t_1[i][0])
        target_q_batch = np.reshape(target_q_batch, [MINI_BATCH_SIZE, 1])
        self.critic_net.train(s_batch, a_batch, target_q_batch)

        # Train learned actor network (Deterministic Policy Gradient theorem is applied here)
        dQ_da_batch = self.critic_net.compute_dQ_da(s_batch, self.actor_net.forward_learned_net(s_batch))
        self.actor_net.train(s_batch, dQ_da_batch)

        # Update target networks making closer to learned networks
        self.critic_net.update_target_net()
        self.actor_net.update_target_net()

