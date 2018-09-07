# 引数にエピソード数入れる

# TODO 変数の読み込み保存


import player11
import threading
import numpy as np
import sys
import os
import random

from ddpg_agent import DDPGAgent, MINI_BATCH_SIZE
from ou_noise import OUNoise


class PlayerDDPG(player11.Player11, threading.Thread):
    def __init__(self):
        super(PlayerDDPG, self).__init__()
        self.m_strCommand = ""
        self.command_step = 0

        # =============for reinforcement learning=================
        # Instanciate reinforcement learning agent which contains Actor/Critic DNN.
        self.num_states = 6
        self.num_actions = 5
        self.actions_max = [3.0, 100.0, 180.0, 100.0, 180.0]
        self.actions_min = [0.0, 0.0, -180.0, 0.0, -180.0]

        self.agent = DDPGAgent(self.num_states, self.num_actions, self.actions_max, self.actions_min)
        # Exploration noise generator which uses Ornstein-Uhlenbeck process.
        self.noise = OUNoise(self.num_actions)

        self.num_this_episode = sys.argv[2]
        # self.step_limit = sys.argv[3]
        self.step_limit = 1000
        self.reward_per_episode = 0
        self.reward_per_step = 0

    def analyzeMessage(self, message):
        """
        メッセージの解析
        :param message:
        :return:
        """
        # 初期メッセージの処理
        # print("p11:message:", message)
        if message.startswith("(init "):
            self.analyzeInitialMessage(message)
        # 視覚メッセージの処理
        elif message.startswith("(see "):
            self.analyzeVisualMessage(message)
        # 体調メッセージの処理
        elif message.startswith("(sense_body "):
            self.analyzePhysicalMessage(message)

            # ====================================================================

            # コマンド実行履歴がある場合の処理
            if self.command_step > 0:
                next_state = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY, self.m_dNeck, self.m_dNeck]
                self.agent.add_experience(self.states, self.actions, next_state, self.reward_per_step, done=False)
                if len(self.agent.replay_buffer) > MINI_BATCH_SIZE: self.agent.train()
                self.reward_per_episode += self.reward_per_step

            self.states = [self.m_dX, self.m_dY, self.m_dBallX, self.m_dBallY, self.m_dNeck, self.m_dNeck]

            # ここがいまいちわからない
            self.actions = self.agent.feed_forward_actor(np.reshape(self.states, [1, self.num_states]))
            self.actions = self.actions[0] + OUNoise(self.num_actions).generate()
            self.play_0()
            self.send(self.m_strCommand)
            if self.m_strPlayMode.startswith("play_on"):
                self.command_step += 1
            self.check_episode_end()
            # ====================================================================

        # 聴覚メッセージの処理
        elif message.startswith("(hear "):
            self.analyzeAuralMessage(message)
        # サーバパラメータの処理
        elif message.startswith("(server_param"):
            self.analyzeServerParam(message)
        # プレーヤーパラメータの処理
        elif message.startswith("(player_param"):
            self.analyzePlayerParam(message)
        # think 処理
        elif message.startswith("(think"):
            self.send("(done)")
        # エラーの処理
        else:
            print("p11 サーバーからエラーが伝えられた:", message)
            print("p11 エラー発生原因のコマンドは右記の通り :", self.m_strCommand)

    # 実行
    def play_0(self):
        """
        コマンドの決定
        :return:
        """
        # キックオフ前？
        if self.checkInitialMode():
            if self.checkInitialMode():
                self.setKickOffPosition()
                command = \
                    "(move " + str(self.m_dKickOffX) + " " + str(self.m_dKickOffY) + ")"
                self.m_strCommand = command
        # (コマンド生成)===================
        if self.actions[0][0] < 1:
            self.m_strCommand = "(dash {})".format(self.actions[1])
        elif self.actions[0][0] < 2:
            self.m_strCommand = "(turn {})".format(self.actions[2])
        elif self.actions[0][0] <= 3:
            self.m_strCommand = "(kick {} {})".format(self.actions[3], self.actions[4])
        # ===============================

    def check_episode_end(self):
        if self.command_step > self.step_limit:
            sys.exit()


if __name__ == "__main__":
    plays = []
    for i in range(22):
        p = PlayerDDPG()
        plays.append(p)
        teamname = str(p.__class__.__name__)
        if i < 11:
            teamname += "left"
        else:
            teamname += "right"
        plays[i].initialize((i % 11 + 1), teamname, "localhost", 6000)
        plays[i].start()

# 離散化させなくてはならない？(6分割**5変数の状態が生み出される)
# 状態s一覧
#
# self.m_dX
# self.m_dY
# self.m_dNeck
# self.m_dBallX
# self.m_dBallY
# self.dStamina
#
# 行動a一覧
# command_slecter
# turn
# dash
# kick
