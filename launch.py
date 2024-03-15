import os
import sys
import argparse
import time

import torch.cuda

from logger.observer import ConsoleLoggerOberver, FileLoggerObserver
from main_processes import ProgramManager
from settings.loader import TrainingSettingLoader

parser = argparse.ArgumentParser(description='2Mode-FlappyBird')


parser.add_argument('-m', '--model', dest='model_path', type=str, default='',
                    help='The name(path) of the model to train or play the game with')

parser.add_argument('-t', '--train', action='store_true', default=False,
                    help='train model instead of playing the game')

parser.add_argument('--cuda', action='store_true', default=False,
                    help='If set true, launch program with cuda enabled; otherwise, with CPU only')


train_argument_group = parser.add_argument_group(
    'Argument for training model (available when you set argument --train)')

train_argument_group.add_argument('--json', type=str,
                                  help='json path to load training setting', default='')
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
# TODO: 目前所有训练episode都会用epsilon-greedy选择action，与help中的解释不符，详见training_setting.exploration在ProgramManager中的用法
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

    program_manager = ProgramManager()

    # 注册日志打印器，之后输出日志时，直接调用父类的打印方法即可
    console_info_logger = ConsoleLoggerOberver()
    console_error_logger = ConsoleLoggerOberver()
    program_manager.register_observer(console_info_logger, 'info')
    program_manager.register_observer(console_error_logger, 'error')

    file_info_logger = FileLoggerObserver('{}_info.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
    file_error_logger = FileLoggerObserver('{}_error.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())))
    program_manager.register_observer(file_info_logger, 'info')
    program_manager.register_observer(file_error_logger, 'error')

    # 如果非训练模式下，模型路径没有被传入程序，则开始真人游玩
    if not args.train and (args.model_path == '' or args.model_path is None):
        program_manager.generate_log(message='argument --model not received, launch game at human mode',
                                     level='info',
                                     location=os.path.split(__file__)[1])
        program_manager.play_game(player='human')
    # 测试cuda是否可用
    elif args.cuda and not torch.cuda.is_available():
        program_manager.generate_log(message='Error: CUDA is not available, maybe you should not set --cuda',
                                     level='error',
                                     location=os.path.split(__file__)[1])
        sys.exit(1)
    # 由模型在游戏环境中游玩或训练
    else:
        program_manager.generate_log(message='launch program with a model given',
                                     level='info',
                                     location=os.path.split(__file__)[1])
        if args.cuda:
            program_manager.generate_log(message='run program with GPU support',
                                         level='info',
                                         location=os.path.split(__file__)[1])
        if args.train:
            """
            从json文件和运行参数中导入设置
            """
            training_setting = TrainingSettingLoader(args, args.json).get_setting()
            # 进入训练过程
            program_manager.train(training_setting)
        else:
            program_manager.play_game(player='computer', args=args)
