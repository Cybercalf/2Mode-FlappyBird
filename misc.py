# TODO: 优化import方式
import PIL.Image as Image
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import random
import numpy as np
import shutil
from BrainDQN import *
import game.dqn_training_gamestate as game_for_training
import game.dqn_mode_gamestate as game_for_playing
import sys
sys.path.append("game/")

# TODO: 将IMAGE_SIZE转移位置，使其不再是全局变量
IMAGE_SIZE = (72, 128)


def preprocess(frame):
    '''
    对输入的帧图像做预处理

    输入图像：512*288，rgb彩色图像

    预处理步骤：

    1.降采样，使图像尺寸变为128*72

    2.将图像转换为灰度图像

    3.将图像每个像素的灰度值映射为0或1
    '''
    im = Image.fromarray(frame).resize(IMAGE_SIZE).convert(mode='L')
    out = np.asarray(im).astype(np.float32)
    out[out <= 1.] = 0.0
    out[out > 1.] = 1.0
    return out


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    '''
    经过一定训练次数后，保存当前模型，并检查其是否是目前最优的模型。

    如果是，则正在训练的模型也会一并更改，接下来将会以它为基础继续训练。

    :param state: checkpoint state: model weight and other info binding by user

    :param is_best: if the checkpoint is the best. If it is, then save as a best model
    '''
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename, model):
    '''
    从磁盘中加载之前保存的检查点模型

    :param filename: model file name

    :param model: DQN model
    '''

    try:
        checkpoint = torch.load(filename)
    # 如果磁盘文件已gpu的方式存储，直接加载到cpu上会出现异常，检测到异常时使用另一种方法加载
    except BaseException:
        # load weight saved on gpy device to cpu device
        # see
        # https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349/3
        checkpoint = torch.load(
            filename, map_location=lambda storage, loc: storage)

    """
    调试输出(debug)
    输出加载的模型在之前经过了多少训练，用于探索的epsilon值为多少
    """
    episode = checkpoint['episode']
    epsilon = checkpoint['epsilon']
    print('pretrained episode = {}'.format(episode))
    print('pretrained epsilon = {}'.format(epsilon))

    model.load_state_dict(checkpoint['state_dict'])

    """
    调试输出(debug)
    输出加载的模型在训练过程中操纵小鸟飞行的最长时间
    """
    time_step = checkpoint.get('best_time_step', None)
    if time_step is None:
        time_step = checkpoint('time_step')
    print('pretrained time step = {}'.format(time_step))

    return episode, epsilon, time_step


def train_dqn(model, options, resume):
    '''
    Train DQN

    :param model: DQN model
    :param options: 训练使用的各项参数
    :param resume: resume previous model

    :param lr: learning rate
    :param max_episode: maximum episode
    :param model_name: checkpoint file name
    '''
    best_time_step = 0.
    if resume:
        if options.weight is None:
            print('[Misc] Error: when resume, you should give weight file name.')
            return
        print('load previous model weight: {}'.format(options.weight))
        _, _, best_time_step = load_checkpoint(options.weight, model)

    """
    初始化各参数
    用于传入action返回状态数据的gamestate
    优化器optimizer
    误差函数ceriterion
    观测数据（这一帧的游戏画面）o
    这一帧的奖励r
    游戏在这一帧是否结束terminal
    """
    flappyBird = game_for_training.GameState()
    optimizer = optim.RMSprop(model.parameters(), lr=options.lr)
    ceriterion = nn.MSELoss()

    action = [1, 0]
    o, r, terminal = flappyBird.frame_step(action)
    o = preprocess(o)
    model.set_initial_state()

    if options.cuda:
        model = model.cuda()

    """
    训练开始前，随机选取action操作小鸟，并将数据保存起来
    随机操作的次数取决于options.observation
    """
    for i in range(options.observation):
        action = model.get_action_randomly()
        o, r, terminal = flappyBird.frame_step(action)
        o = preprocess(o)
        model.store_transition(o, action, r, terminal)

    # start training
    for episode in range(options.max_episode):
        model.time_step = 0
        model.set_train()
        total_reward = 0.

        # ------beginning of an episode------
        while True:
            optimizer.zero_grad()

            # 模型依据自身经验决定这一帧采取的action，传入gamestate，获得这一帧的观测图像、奖励值、游戏是否中止
            action = model.get_action()
            o_next, r, terminal = flappyBird.frame_step(action)
            total_reward += options.gamma**model.time_step * r

            # 对这一帧图像做预处理
            o_next = preprocess(o_next)

            # 保存数据，模型的time_step计数器+1
            model.store_transition(o_next, action, r, terminal)
            model.increase_time_step()

            # Step 1: obtain random minibatch from replay memory
            minibatch = random.sample(model.replay_memory, options.batch_size)
            state_batch = np.array([data[0] for data in minibatch])
            action_batch = np.array([data[1] for data in minibatch])
            reward_batch = np.array([data[2] for data in minibatch])
            next_state_batch = np.array([data[3] for data in minibatch])
            state_batch_var = Variable(torch.from_numpy(state_batch))
            with torch.no_grad():
                next_state_batch_var = Variable(
                    torch.from_numpy(next_state_batch))
            if options.cuda:
                state_batch_var = state_batch_var.cuda()
                next_state_batch_var = next_state_batch_var.cuda()

            """
            Step 2: calculate y
            TODO: 看懂y值的计算步骤
            """
            q_value_next = model.forward(next_state_batch_var)

            q_value = model.forward(state_batch_var)

            y = reward_batch.astype(np.float32)
            max_q, _ = torch.max(q_value_next, dim=1)

            for i in range(options.batch_size):
                if not minibatch[i][4]:
                    y[i] += options.gamma * max_q.data[i].item()

            y = Variable(torch.from_numpy(y))
            action_batch_var = Variable(torch.from_numpy(action_batch))
            if options.cuda:
                y = y.cuda()
                action_batch_var = action_batch_var.cuda()
            q_value = torch.sum(torch.mul(action_batch_var, q_value), dim=1)

            loss = ceriterion(q_value, y)
            loss.backward()

            optimizer.step()
            # when the bird dies, the episode ends
            if terminal:
                break

        # ------end of an episode------

        print('episode: {}, epsilon: {:.4f}, max time step: {}, total reward: {:.6f}'.format(
            episode, model.epsilon, model.time_step, total_reward))

        # 经过一次episode后，降低epsilon的值
        if model.epsilon > options.final_e:
            delta = (options.init_e - options.final_e) / options.exploration
            model.epsilon -= delta

        """
        每经过100次episode，测试训练后模型的效果
        如果训练后的模型效果优于训练前的模型，将其保存起来
        否则，按照options.save_checkpoint_freq的值，每隔一定数量的episode保存一次模型，不管这个模型是否是当前最优的
        TODO: 使测试间隔的episode数量不再用硬编码表示
        TODO: 使每个checkpoint文件的state都同时包含time_step和best_time_step
        """
        if episode % 100 == 0:
            ave_time = test_dqn(model, episode)

        if ave_time > best_time_step:
            best_time_step = ave_time
            save_checkpoint({
                'episode': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'best_time_step': best_time_step,
            }, True, 'checkpoint-episode-%d.pth.tar' % episode)
        elif episode % options.save_checkpoint_freq == 0:
            save_checkpoint({
                'episode:': episode,
                'epsilon': model.epsilon,
                'state_dict': model.state_dict(),
                'time_step': ave_time,
            }, False, 'checkpoint-episode-%d.pth.tar' % episode)
        else:
            continue
        print('save checkpoint, episode={}, ave time step={:.2f}'.format(
            episode, ave_time))


def test_dqn(model, episode):
    '''
    测试当前模型的游戏效果

    具体步骤：

    模型在与训练相同的gamestate下游玩5次，游玩过程采取的action全部为依据自身数据选择的，而非随机选取。5次游戏结束后，返回平均游戏时间。

    :param model: dqn model
    :param episode: current training episode
    :returns ave_time: 模型在5次游戏中坚持的平均时间
    '''
    # TODO: 使游戏次数不再用硬编码表示
    model.set_eval()
    ave_time = 0.
    for test_case in range(5):
        model.time_step = 0
        flappyBird = game_for_training.GameState()
        o, r, terminal = flappyBird.frame_step([1, 0])
        o = preprocess(o)
        model.set_initial_state()
        while True:
            action = model.get_optim_action()
            o, r, terminal = flappyBird.frame_step(action)
            if terminal:
                break
            o = preprocess(o)
            model.current_state = np.append(
                model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)
            model.increase_time_step()
        ave_time += model.time_step
    ave_time /= 5
    print('testing: episode: {}, average time: {}'.format(episode, ave_time))
    return ave_time


def play_game(model_file_name, cuda=False, best=True):
    '''
    使用一个训练过的模型玩flappybird游戏

    :param weight: model file name containing weight of dqn
    :param best: if the model is best or not
    '''

    # 调试输出
    print('load pretrained model file: ' + model_file_name)
    model = BrainDQN(epsilon=0., mem_size=0, cuda=cuda)
    load_checkpoint(model_file_name, model)

    model.set_eval()
    bird_game = game_for_playing.GameState()
    model.set_initial_state()
    if cuda:
        model = model.cuda()
    while True:
        action = model.get_optim_action()
        o, r, terminal = bird_game.frame_step(action)
        if terminal:
            break
        o = preprocess(o)

        model.current_state = np.append(
            model.current_state[1:, :, :], o.reshape((1,) + o.shape), axis=0)

        model.increase_time_step()
    print('total time step is {}'.format(model.time_step))
