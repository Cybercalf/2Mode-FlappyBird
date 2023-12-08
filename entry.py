import sys
import argparse
import os
from game.human_mode_gamestate import FlappyBirdGameManager

parser = argparse.ArgumentParser(description='2Mode-FlappyBird')

parser.add_argument('--mode', type=str, help='2 mode supported: train / play', default='play')

parser.add_argument('--play_mode', type=str, help='the way you want to play the game', default='human')

if __name__ == '__main__':
    '''
    解析传入参数，针对部分情况给出错误信息
    '''
    args = parser.parse_args()
    # 游玩模式
    if args.mode == 'play':
        # 人类游玩
        if args.play_mode == "human":
            print('[Entry] Start game at human mode')
            game = FlappyBirdGameManager()
            game.game_start(with_frame_step=True)
        # 让训练好的模型游玩
        elif args.play_mode == "dqn":
            print('[Entry] Start game at dqn mode')
            os.system('python main.py --weight model_best.pth.tar --cuda')
        # 无效的游戏模式
        else:
            print('[Entry] Error: invalid game mode')
            sys.exit(1)
