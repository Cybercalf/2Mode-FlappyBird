# 2Modes-FlappyBird

`2Modes-FlappyBird` is a mini project that allows people to play FlappyBird by themselves, or by computers through a model based on Reinforcement Learning. You can train models in a DQN framework by writing training preset (json) and executing commands in console, or you can write your own training scripts based on a game environment compliant with [Gymnasium](https://gymnasium.farama.org/).

---

## Dependencies

Here are the versions of some of the dependencies I used:

```yaml
dependencies:
  - python=3.11.5
  - pip:
      - gymnasium==0.29.1
      - numpy==1.26.0
      - pygame==2.5.2
      - stable-baselines3==2.2.1
      - tensorboard==2.16.2
      - torch==2.1.0+cu121
      - torchaudio==2.1.0+cu121
      - torchvision==0.16.0+cu121
```

[This guide](https://blog.csdn.net/weixin_42634080/article/details/125360470) provides the complete precess of installing `PyTorch` and `CUDA`. To install other necessary dependencies, you can simply type `pip install stable-baselines3[extra]`. More information in [official installation guide](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).

---

## User Guide

There are 2 ways to train your own model currently:

### 1. Working under Gymnasium

`flappybird.env` provides a custom game environment compliant with [Gymnasium](https://gymnasium.farama.org/). When you import module `flappybird`, it will be automatically registered.

|||
|---|---|
|Action Space|`Discrete(2)`|
|Observation Space|`Box(low=0, high=255, shape=(4, 128, 72), dtype=np.uint8)`|
|import|`gymnasium.make("FlappyBird-v0")`|

action space:

|Num|Action|
|---|---|
|0|Don't do anything. Let the bird move with inertia and gravity.|
|1|Flap wings once.|

observation space:

Four 128x72 grayscale image of the bird and pipes within the screen, representing 4 consecutive frames of the game, where the last frame is the latest.

Writing a script under [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) framework is an ideal way to train your own model. Please read their [user guide]((https://stable-baselines3.readthedocs.io/en/master/)) for more information.

### 2. Custom Framework

With a console, you can also simply type commands to train a model, evaluate a model or just enjoy the game by yourself.

```shell
# There are many optional parameters to help you train a model. To know all the the parameters and what they do, type:
python launch.py -h
# To play game by yourself, simply go into the root of the project and type:
python launch.py
# To play game with a pretrained model, type:
python launch.py --model <name of the model you want to play the game with>
# To train a model from scratch with GPU support, type:
python launch.py --train --cuda
```

You can write the training presets in a json file before the training begins. Conforming json contents looks like:

```json
{
    "lr": 0.0001,
    "gamma": 0.99,
    "batch_size": 72,
    "memory_size": 5000,
    "observation": 100,
    "max_episode": 9000,
    "resume": false,
    "test_model_freq": 100,
    "save_checkpoint_freq": 1000,
    "update_target_qnetwork_freq": 10,
    "exploration": 9000,
    "exploration_method": "Epsilon Greedy",
    "epsilon_greedy": {
        "init_e": 1.0,
        "final_e": 0.1
    },
    "boltzmann_exploration": {
        "tau": 1.0
    },
    "advanced_method": ["Double DQN", "Dueling DQN"]
}
```

After that, you can start the training using your own preset:

```shell
python launch.py --json <name of your preset file>.json
```

Please note that if you launch the project via `launch.py`, the training process will use CPU by default. To transfrom your model to GPU, add parameter `--cuda` to your command explicitly.

---

### Disclaimer

The structure of classic Q-Network in this project currently references [xmfbit/DQN-FlappyBird](https://github.com/xmfbit/DQN-FlappyBird), which is based on other contributors' projects.
