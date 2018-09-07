# ライブラリインポート
import numpy as np
import gym

ENV_NAME = "InvertedPendulum-v2"

env = gym.make(ENV_NAME)

print(env.action_space)
