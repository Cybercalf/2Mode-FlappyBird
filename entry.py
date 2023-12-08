import sys
import argparse
import os
from game.human_mode_gamestate import FlappyBirdGameManager

parser = argparse.ArgumentParser(description='2Mode-FlappyBird')

parser.add_argument('--mode', type=str, help='game mode', default='human')

if __name__ == '__main__':
    '''
    解析传入参数，针对部分情况给出错误信息
    '''
    args = parser.parse_args()
    if args.mode == "human":
        print('human mode')
        game = FlappyBirdGameManager()
        game.game_start()
    elif args.mode == "dqn":
        print('dqn mode')
        os.system('python main.py --weight model_best.pth.tar --cuda')
    else:
        print('Error: invalid game mode')
        sys.exit(1)
