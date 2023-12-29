import sys
import argparse
import main_processes
import torch.cuda

parser = argparse.ArgumentParser(description='2Mode-FlappyBird')


parser.add_argument('-m', '--model', dest='model_path', type=str, default='',
                    help='The name(path) of the model to train or play the game with')

parser.add_argument('-t', '--train', action='store_true', default=False,
                    help='train model instead of playing the game')

parser.add_argument('--cuda', action='store_true', default=False,
                    help='If set true, launch program with cuda enabled; otherwise, with CPU only')


train_argument_group = parser.add_argument_group(
    'Argument for training model (available when you set argument --train)')

train_argument_group.add_argument('--lr', type=float,
                                  help='learning rate', default=0.0001)
train_argument_group.add_argument('--gamma', type=float,
                                  help='discount rate', default=0.99)
train_argument_group.add_argument('--batch_size', type=int,
                                  help='batch size', default=32)
train_argument_group.add_argument('--memory_size', type=int,
                                  help='memory size for experience replay', default=5000)
train_argument_group.add_argument('--init_e', type=float,
                                  help='initial epsilon for epsilon-greedy exploration',
                                  default=1.0)
train_argument_group.add_argument('--final_e', type=float,
                                  help='final epsilon for epsilon-greedy exploration',
                                  default=0.1)
train_argument_group.add_argument('--observation', type=int,
                                  help='random observation number in the beginning before training',
                                  default=100)
train_argument_group.add_argument('--exploration', type=int,
                                  help='number of exploration using epsilon-greedy policy',
                                  default=10000)
train_argument_group.add_argument('--max_episode', type=int,
                                  help='maximum episode of training',
                                  default=20000)
train_argument_group.add_argument('--resume', action='store_true', default=False,
                                  help='whether to start training based on model given (finetuning model)',)
train_argument_group.add_argument('--test_model_freq', type=int,
                                  help='episode interval to test model during training phase', default=100)
train_argument_group.add_argument('--save_checkpoint_freq', type=int,
                                  help='episode interval to save checkpoint', default=2000)


if __name__ == '__main__':
    '''
    解析传入参数，针对部分情况给出错误信息
    '''
    args = parser.parse_args()

    # 如果模型路径没有被传入程序，则开始真人游玩
    if args.model_path == '' or args.model_path is None:
        print('[launch.py] argument --model not received, launch game at human mode')
        main_processes.play_game(player='human')
    # 测试cuda是否可用
    elif args.cuda and not torch.cuda.is_available():
        print(
            '[launch.py] Error: CUDA is not available, maybe you should not set --cuda')
        sys.exit(1)
    # 由模型在游戏环境中游玩或训练
    else:
        print('[launch.py] launch program with a model given')
        if args.cuda:
            print('[launch.py] run program with GPU support')
        if args.train:
            main_processes.train_model(args)
        else:
            main_processes.play_game(player='computer', args=args)
