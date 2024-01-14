import json


class TrainingSettingLoader:
    '''
    从json文件和程序运行参数中导入训练设置
    '''

    def __init__(self, args=None, json_path=''):
        # 将设置初始化为默认设置内容
        self.setting = DefaultTrainingSetting()
        # 从运行参数中加载设置
        self.setting_from_args = args
        # 从json文件中加载设置
        self.setting_from_json = JsonLoader(json_path).get_content()
        # 验证加载内容的有效性
        # 导入设置的优先级：json > 运行参数
        if self.setting_from_args is not None:
            self.validate_setting_from_args(self.setting_from_args)
        if self.setting_from_json is not None:
            self.validate_setting_from_json(self.setting_from_json)

    def get_setting(self):
        return self.setting

    def validate_setting_from_json(self, json_dict):
        '''
        验证从json导入的设置内容的有效性，如果有效，用它覆盖原来的设置内容
        '''

        lr = json_dict.get('lr', None)
        if lr:
            if 0 < lr:
                self.setting.lr = lr

        gamma = json_dict.get('gamma', None)
        if gamma:
            if 0 <= gamma <= 1:
                self.setting.gamma = gamma

        batch_size = json_dict.get('batch_size', None)
        if batch_size:
            if 0 < batch_size:
                self.setting.batch_size = batch_size

        memory_size = json_dict.get('memory_size', None)
        if memory_size:
            if 0 < memory_size:
                self.setting.memory_size = memory_size

        observation = json_dict.get('observation', None)
        if observation:
            if 0 < observation:
                self.setting.observation = observation

        max_episode = json_dict.get('max_episode', None)
        if max_episode:
            if 0 < max_episode:
                self.setting.max_episode = max_episode

        resume = json_dict.get('resume', None)
        if isinstance(resume, bool):
            self.setting.resume = resume

        test_model_freq = json_dict.get('test_model_freq', None)
        if test_model_freq:
            if 0 < test_model_freq:
                self.setting.test_model_freq = test_model_freq

        save_checkpoint_freq = json_dict.get('save_checkpoint_freq', None)
        if save_checkpoint_freq:
            if 0 < save_checkpoint_freq:
                self.setting.save_checkpoint_freq = save_checkpoint_freq

        update_target_qnetwork_freq = json_dict.get('update_target_qnetwork_freq', None)
        if update_target_qnetwork_freq:
            if 0 < update_target_qnetwork_freq:
                self.setting.update_target_qnetwork_freq = update_target_qnetwork_freq

        exploration = json_dict.get('exploration', None)
        if exploration:
            if 0 < exploration:
                self.setting.exploration = exploration

        exploration_method = json_dict.get('exploration_method', None)
        if exploration_method in ['Epsilon Greedy', 'Boltzmann Exploration']:
            self.setting.exploration_method = exploration_method

        epsilon_greedy = json_dict.get('epsilon_greedy', None)
        if isinstance(epsilon_greedy, dict):
            init_e = epsilon_greedy.get('init_e', None)
            final_e = epsilon_greedy.get('final_e', None)
            if init_e and final_e:
                if 0 <= final_e <= init_e <= 1:
                    self.setting.epsilon_greedy.init_e = init_e
                    self.setting.epsilon_greedy.final_e = final_e

        boltzmann_exploration = json_dict.get('boltzmann_exploration', None)
        if isinstance(boltzmann_exploration, dict):
            tau = boltzmann_exploration.get('tau', None)
            if tau:
                if tau > 0:
                    self.setting.boltzmann_exploration.tau = tau

        # TODO: 后续可能要修改advanced_method的结构，使多种进阶技术可以被使用，目前一次最多只能使用一种
        # TODO: 尝试引入Prioritized Reply
        advanced_method = json_dict.get('advanced_method', None)
        if advanced_method in ['None', 'Double DQN', 'Dueling DQN', 'Prioritized Replay', 'Multi-step', 'Noisy Net']:
            self.setting.advanced_method = advanced_method

    def validate_setting_from_args(self, args):
        '''
        验证从运行参数导入的设置内容的有效性，如果有效，用它覆盖原来的设置内容
        '''

        lr = args.lr
        if lr:
            if 0 < lr:
                self.setting.lr = lr

        gamma = args.gamma
        if gamma:
            if 0 <= gamma <= 1:
                self.setting.gamma = gamma

        batch_size = args.batch_size
        if batch_size:
            if 0 < batch_size:
                self.setting.batch_size = batch_size

        memory_size = args.memory_size
        if memory_size:
            if 0 < memory_size:
                self.setting.memory_size = memory_size

        observation = args.observation
        if observation:
            if 0 < observation:
                self.setting.observation = observation

        max_episode = args.max_episode
        if max_episode:
            if 0 < max_episode:
                self.setting.max_episode = max_episode

        resume = args.resume
        if isinstance(resume, bool):
            self.setting.resume = resume

        test_model_freq = args.test_model_freq
        if test_model_freq:
            if 0 < test_model_freq:
                self.setting.test_model_freq = test_model_freq

        save_checkpoint_freq = args.save_checkpoint_freq
        if save_checkpoint_freq:
            if 0 < save_checkpoint_freq:
                self.setting.save_checkpoint_freq = save_checkpoint_freq

        exploration = args.exploration
        if exploration:
            if 0 < exploration:
                self.setting.exploration = exploration

        # exploration_method = setting_to_validate.get('exploration_method', None)
        # if exploration_method in ['Epsilon Greedy', 'Boltzmann Exploration']:
        #     self.setting.exploration_method = exploration_method

        init_e = args.init_e
        final_e = args.final_e
        if init_e and final_e:
            if 0 <= final_e <= init_e <= 1:
                self.setting.epsilon_greedy.init_e = init_e
                self.setting.epsilon_greedy.final_e = final_e


class DefaultTrainingSetting:
    '''
    默认的训练设置
    '''

    class EpsilonGreedy:
        def __init__(self):
            self.init_e = 1.0
            self.final_e = 0.1

    class BoltzmannExploration:
        def __init__(self):
            self.tau = 1.0

    def __init__(self):
        self.lr = 0.0001
        self.gamma = 0.99
        self.batch_size = 32
        self.memory_size = 5000
        self.observation = 100
        self.max_episode = 20000
        self.resume = False
        self.test_model_freq = 100
        self.save_checkpoint_freq = 2000
        self.update_target_qnetwork_freq = 10
        self.exploration = 10000
        self.exploration_method = 'Epsilon Greedy'
        self.epsilon_greedy = self.EpsilonGreedy()
        self.boltzmann_exploration = self.BoltzmannExploration()
        self.advanced_method = 'None'


class JsonLoader:
    '''
    从json文件中导入内容
    '''

    def __init__(self, json_path):
        try:
            with open(json_path, 'r') as f:
                self.content = json.load(f)
        except Exception:
            self.content = None
            raise FileNotFoundError('json not found')

    def get_content(self):
        return self.content
