import subprocess
import os
import time

# エピソード数
episodes = 10000
# ステップ数
step = 1000
# 実行ファイル名
exe_program = "playerDDPG.py"
# 状態空間次元数
state_num = 6
# 行動空間次元数
action_num = 1

if __name__ == "__main__":

    # 本番学習スタート
    print("start")
    for episode in range(episodes):
        # ディレクトリの移動
        os.chdir("../")
        os.chdir("../")

        # サーバの起動
        cmd = \
            "rcssserver server::half_time = -1 server::send_step = 3 server::sense_body_step = 2 server::simulator_step = 2 server::auto_mode = true server::kick_off_wait = 200"
        server = subprocess.Popen(cmd.split())

        # モニタの起動
        cmd = "soccerwindow2"
        window = subprocess.Popen(cmd.split())

        # ディレクトリの移動
        os.chdir("./AgentDDPG/src")

        if not os.path.isdir("./models"):
            os.mkdir("./models")

        if not os.path.isdir("./logs"):
            os.mkdir("./logs")

        # クライアントプログラムの実行
        cmd = "python3 {} {}, {}".format(exe_program, episode, step)
        cliant = subprocess.Popen(cmd.split())

        # 学習
        # while True:
        #     if zidan2.episode_finish_flag is True:
        #         break
        time.sleep(15)
        print("episode{} is done ".format(episode))

        # サーバの削除
        server.kill()
        # ウィンドウの削除
        window.kill()
        # クライアントの削除
        cliant.kill()

    print("end")
