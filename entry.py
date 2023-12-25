import sys
import argparse
import misc
import BrainDQN
import torch.cuda
from game.human_mode_gamestate import FlappyBirdGameManager

parser = argparse.ArgumentParser(description='2Mode-FlappyBird')

parser.add_argument('--mode', type=str,
                    help='2 mode supported: train / play', default='play')

parser.add_argument('--play_mode', type=str,
                    help='(Only useful when mode==play) The way you want to play the game', default='human')
parser.add_argument('--model', type=str, default='',
                    help='(Only useful when mode==play) The name of model file you want to play the game with')

parser.add_argument('--cuda', action='store_true', default=False,
                    help='(Only useful when mode==train) If set true, with cuda enabled; otherwise, with CPU only')
parser.add_argument('--lr', type=float,
                    help='(Only useful when mode==train) learning rate', default=0.0001)
parser.add_argument('--gamma', type=float,
                    help='(Only useful when mode==train) discount rate', default=0.99)
parser.add_argument('--batch_size', type=int,
                    help='(Only useful when mode==train) batch size', default=32)
parser.add_argument('--memory_size', type=int,
                    help='(Only useful when mode==train) memory size for experience replay', default=5000)
parser.add_argument('--init_e', type=float,
                    help='(Only useful when mode==train) initial epsilon for epsilon-greedy exploration',
                    default=1.0)
parser.add_argument('--final_e', type=float,
                    help='(Only useful when mode==train) final epsilon for epsilon-greedy exploration',
                    default=0.1)
parser.add_argument('--observation', type=int,
                    help='(Only useful when mode==train) random observation number in the beginning before training',
                    default=100)
parser.add_argument('--exploration', type=int,
                    help='(Only useful when mode==train) number of exploration using epsilon-greedy policy',
                    default=10000)
parser.add_argument('--max_episode', type=int,
                    help='(Only useful when mode==train) maximum episode of training',
                    default=20000)
parser.add_argument('--weight', type=str,
                    help='(Only useful when mode==train) weight file name for finetunig(Optional)', default='')
parser.add_argument('--save_checkpoint_freq', type=int,
                    help='(Only useful when mode==train) episode interval to save checkpoint', default=2000)

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
            # 非训练环境。必须给定一个预训练模型。
            if not args.model == '':
                print('[Entry] Start game at dqn mode')
                misc.play_game(args.model, args.cuda, True)
            else:
                print(
                    '[Entry] Error: When test (simply play game with model), a pretrained weight model file should be given')
                sys.exit(1)
        # 无效的游戏模式
        else:
            print('[Entry] Error: invalid game mode')
            sys.exit(1)
    # 训练模式
    elif args.mode == 'train':
        if args.cuda and not torch.cuda.is_available():
            print(
                '[Entry] Error: CUDA is not available, maybe you should not set --cuda')
            sys.exit(1)
        if args.cuda:
            print('[Entry] Train model with GPU support')
        else:
            print('[Entry] Train model with CPU')
        # 训练环境。如果给定了预训练模型，则在指定模型的基础上继续训练，否则从头开始训练一个模型
        model = BrainDQN.BrainDQN(epsilon=args.init_e,
                                  mem_size=args.memory_size, cuda=args.cuda)
        resume = not args.weight == ''
        misc.train_dqn(model, args, resume)
